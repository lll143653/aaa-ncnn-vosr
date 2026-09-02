#!/usr/bin/env python
"""One-click VOSR2 -> ncnn conversion.

Produces five ncnn models from the released VOSR2 checkpoint:
  <out>/dinov2l_feat.ncnn.{param,bin}   DINOv2-L layer-17 feature extractor
  <out>/vosr_dit_a.ncnn.{param,bin}     DiT blocks 0..17: (lq, noise, z) -> tokens
  <out>/vosr_dit_b.ncnn.{param,bin}     DiT blocks 18..35 + final: tokens -> velocity tokens
  <out>/vae_encoder.ncnn.{param,bin}    Qwen-Image 2D VAE encoder (normalized latent)
  <out>/vae_decoder.ncnn.{param,bin}    Qwen-Image 2D VAE decoder (pixels [-1,1])

The DiT is split into two chained halves because pnnx's pass_ncnn crashes on
the full 5.5GB fp32 graph; the two ~2.7GB halves convert fine.

All graphs are fixed-shape: 512x512 pixel tile / 64x64 latent / 32x32 tokens.

Example:
  python scripts/convert.py \
      --repo /path/to/VOSR-main \
      --checkpoint preset/ckpts/VOSR2 \
      --output models
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="path to the VOSR repo")
    p.add_argument("--checkpoint", required=True, help="VOSR2 checkpoint dir (with args.json)")
    p.add_argument("--output", required=True, help="output dir for ncnn models")
    p.add_argument("--pnnx-exe", default=shutil.which("pnnx"), help="pnnx executable")
    p.add_argument("--skip-check", action="store_true", help="skip torch parity check")
    p.add_argument("--skip-validate", action="store_true", help="skip ncnn smoke validation")
    return p.parse_args()


def load_vosr(repo, ckpt_dir):
    """Load VAE + DINOv2-L + DiT exactly like inference_vosr_onestep.py."""
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)  # torch.hub.set_dir uses a relative path

    torch.hub.set_dir("preset/ckpts/torch_cache")

    with open(Path(ckpt_dir) / "args.json") as f:
        cfg = json.load(f)

    from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
    from models.lightningdit import LightningDiT
    from safetensors.torch import load_file

    vae = AutoencoderKLQwenImage2D.from_pretrained(cfg["ae_path"]).eval()

    encoder = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vitl14", skip_validation=True)
    del encoder.head
    encoder.head = torch.nn.Identity()
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    model = LightningDiT(
        input_size=cfg["resolution"] // 8,
        patch_size=cfg["patch_size"],
        in_channels=2 * 16,
        out_channels=16,
        hidden_size=cfg["dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        z_dims=cfg["enc_dim"],
        encdim_ratio=cfg["encdim_ratio"],
        auxiliary_time_cond=bool(cfg.get("auxiliary_time_cond", False)),
        use_qknorm=cfg["use_qknorm"],
        use_swiglu=cfg["use_swiglu"],
        use_rope=cfg["use_rope"],
        use_rmsnorm=cfg["use_rmsnorm"],
        wo_shift=cfg["wo_shift"],
        num_fused_layers=len(cfg["layer_dinov2b_list"]),
    )
    state = load_file(str(Path(ckpt_dir) / "checkpoints" / "ema_model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"DiT weights: {len(missing)} missing, {len(unexpected)} unexpected")
    model.eval()
    return vae, encoder, model, cfg


def main():
    args = parse_args()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.pnnx_exe:
        sys.exit("pnnx not found on PATH; activate the env (pip install pnnx) "
                 "or pass --pnnx-exe")

    from vosr_ncnn import (
        DeployDiT, DinoV2FeatExport, VaeEncoderExport, VaeDecoderExport,
        patch_sdpa, restore_sdpa, unpatchify_tokens,
    )

    print("== loading checkpoint ==", flush=True)
    vae, encoder, model, cfg = load_vosr(args.repo, args.checkpoint)
    grid = cfg["resolution"] // 8 // cfg["patch_size"]   # 512 -> 64 -> 32
    dino_size = cfg["dinov2_size"]                        # 448
    n_tokens = (dino_size // 14) ** 2                     # 1024
    enc_dim = cfg["enc_dim"]                              # 1024
    print(f"grid={grid} dino_input={dino_size} dino_tokens={n_tokens}", flush=True)

    dino = DinoV2FeatExport(encoder, layer_id=cfg["layer_dinov2b_list"][0]).eval()
    deploy = DeployDiT(model, grid=grid).eval()
    # pnnx's pass_ncnn crashes on the full 5.5GB fp32 DiT graph (protobuf-free
    # build too), so the DiT is converted as two chained halves.
    half = cfg["depth"] // 2
    deploy_a = DeployDiT(model, grid=grid, block_range=(0, half), tail=False).eval()
    deploy_b = DeployDiT(model, grid=grid, block_range=(half, cfg["depth"]), head=False).eval()
    venc = VaeEncoderExport(vae).eval()
    vdec = VaeDecoderExport(vae).eval()

    torch.manual_seed(0)
    lat_hw = grid * cfg["patch_size"]                     # 64
    res = lat_hw * 8                                      # 512
    lq_latent = torch.randn(1, 16, lat_hw, lat_hw)
    noise = torch.randn(1, 16, lat_hw, lat_hw)
    z_raw = torch.randn(1, n_tokens, enc_dim)
    dino_in = torch.randn(1, 3, dino_size, dino_size)
    pix = torch.randn(1, 3, res, res).clamp(-1, 1)

    # ---- torch parity check (deploy graph vs stock modules) ----
    if not args.skip_check:
        print("== torch parity check ==", flush=True)
        with torch.no_grad():
            # DiT: deploy tokens vs stock one-step velocity
            ref_u = model(torch.cat([lq_latent, noise], 1),
                          torch.ones(1), torch.zeros(1), z=[z_raw])
            dep_u = unpatchify_tokens(deploy(lq_latent, noise, z_raw), grid,
                                      cfg["patch_size"], 16)
            print(f"dit:        max abs diff = {(ref_u - dep_u).abs().max().item():.3e}", flush=True)

            # DINOv2: export vs forward_with_features path
            feats = {}
            x = encoder.prepare_tokens_with_masks(dino_in, None)
            for i, blk in enumerate(encoder.blocks):
                x = blk(x)
                feats[i] = x[:, 1:]
            ref_f = feats[cfg["layer_dinov2b_list"][0]]
            dep_f = dino(dino_in)
            print(f"dinov2:     max abs diff = {(ref_f - dep_f).abs().max().item():.3e}", flush=True)

            # VAE encoder: export vs encode+mode+normalize
            post = vae.encode(pix).latent_dist
            lm = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)
            ls = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
            ref_lat = (post.mode() - lm) * ls
            dep_lat = venc(pix)
            print(f"vae_enc:    max abs diff = {(ref_lat - dep_lat).abs().max().item():.3e}", flush=True)

            # VAE decoder
            ref_pix = vae.decode(ref_lat / ls + lm, return_dict=False)[0].clamp(-1, 1)
            dep_pix = vdec(ref_lat)
            print(f"vae_dec:    max abs diff = {(ref_pix - dep_pix).abs().max().item():.3e}", flush=True)

    # ---- trace ----
    # Decompose SDPA into matmul+softmax in all graphs: ncnn has no portable
    # SDPA layer across versions (custom-layer registration), while the
    # decomposed form works everywhere.
    print("== tracing ==", flush=True)
    orig = patch_sdpa()
    try:
        with torch.no_grad():
            ref_f = dino(dino_in)
            torch.jit.trace(dino, dino_in).save(str(out_dir / "dinov2l_feat.pt"))
            print("  dinov2l_feat.pt", flush=True)
            ref_tokens = deploy(lq_latent, noise, z_raw)
            mid_tokens = deploy_a(lq_latent, noise, z_raw)
            d = (deploy_b(mid_tokens, z_raw) - ref_tokens).abs().max().item()
            print(f"  dit a|b chain vs full: max abs diff = {d:.3e}", flush=True)
            torch.jit.trace(deploy_a, (lq_latent, noise, z_raw)).save(str(out_dir / "vosr_dit_a.pt"))
            print("  vosr_dit_a.pt", flush=True)
            torch.jit.trace(deploy_b, (mid_tokens, z_raw)).save(str(out_dir / "vosr_dit_b.pt"))
            print("  vosr_dit_b.pt", flush=True)
            ref_lat = venc(pix)
            torch.jit.trace(venc, pix).save(str(out_dir / "vae_encoder.pt"))
            print("  vae_encoder.pt", flush=True)
            ref_pix = vdec(ref_lat)
            torch.jit.trace(vdec, ref_lat).save(str(out_dir / "vae_decoder.pt"))
            print("  vae_decoder.pt", flush=True)
    finally:
        restore_sdpa(orig)

    # ---- pnnx ----
    dim = cfg["dim"]
    conv_specs = [
        ("dinov2l_feat", f"[1,3,{dino_size},{dino_size}]f32"),
        ("vosr_dit_a", f"[1,16,{lat_hw},{lat_hw}]f32,[1,16,{lat_hw},{lat_hw}]f32,[1,{n_tokens},{enc_dim}]f32"),
        ("vosr_dit_b", f"[1,{n_tokens},{dim}]f32,[1,{n_tokens},{enc_dim}]f32"),
        ("vae_encoder", f"[1,3,{res},{res}]f32"),
        ("vae_decoder", f"[1,16,{lat_hw},{lat_hw}]f32"),
    ]
    for name, shape in conv_specs:
        print(f"== pnnx {name} ==", flush=True)
        r = subprocess.run(
            [args.pnnx_exe, str(out_dir / f"{name}.pt"), f"inputshape={shape}", "fp16=0"],
            cwd=out_dir, capture_output=True, text=True,
        )
        log = r.stdout + r.stderr
        for l in log.splitlines():
            if "not supported" in l or "ignore" in l:
                print(f"  [warn] {l}", flush=True)
        produced = (out_dir / f"{name}.ncnn.param").is_file() and (out_dir / f"{name}.ncnn.bin").is_file()
        if r.returncode != 0 or not produced:
            print(log)
            sys.exit(f"pnnx failed for {name}")
        for junk in out_dir.glob(f"{name}.pnnx.*"):
            junk.unlink()
        for junk in (out_dir / f"{name}_pnnx.py", out_dir / f"{name}_ncnn.py",
                     out_dir / f"{name}.pt"):
            if junk.exists():
                junk.unlink()
        print("  ok", flush=True)

    # ---- ncnn smoke validation ----
    if not args.skip_validate:
        try:
            import ncnn
            import numpy as np
        except ImportError:
            print("ncnn python not installed; skipping smoke validation")
            return
        print("== ncnn smoke validation (CPU) ==", flush=True)

        def run(name, inputs):
            net = ncnn.Net()
            net.opt.num_threads = 8
            assert net.load_param(str(out_dir / f"{name}.ncnn.param")) == 0
            assert net.load_model(str(out_dir / f"{name}.ncnn.bin")) == 0
            ex = net.create_extractor()
            keepalive = []
            for i, arr in enumerate(inputs):
                # ncnn-allocated aligned Mat (SIMD kernels assume 32B alignment;
                # numpy buffers are not guaranteed aligned -> copy in)
                arr = np.ascontiguousarray(arr, dtype=np.float32)
                if arr.ndim == 3:
                    m = ncnn.Mat(arr.shape[2], arr.shape[1], arr.shape[0])
                elif arr.ndim == 2:
                    m = ncnn.Mat(arr.shape[1], arr.shape[0])
                else:
                    raise ValueError(arr.shape)
                np.array(m, copy=False)[:] = arr
                keepalive.append(m)
                ex.input(f"in{i}", m)
            ret, out = ex.extract("out0")
            assert ret == 0
            return np.array(out, dtype=np.float32, copy=True)

        n_ncnn = run("dinov2l_feat", [dino_in[0].numpy()])
        d = np.abs(n_ncnn.reshape(ref_f.shape[1:]) - ref_f[0].numpy())
        print(f"dinov2l_feat: max abs diff = {d.max():.3e}", flush=True)

        mid_ncnn = run("vosr_dit_a", [lq_latent[0].numpy(), noise[0].numpy(), z_raw[0].numpy()])
        t_ncnn = run("vosr_dit_b", [mid_ncnn, z_raw[0].numpy()])
        d = np.abs(t_ncnn.reshape(ref_tokens.shape[1:]) - ref_tokens[0].numpy())
        print(f"vosr_dit a|b: max abs diff = {d.max():.3e}", flush=True)

        l_ncnn = run("vae_encoder", [pix[0].numpy()])
        d = np.abs(l_ncnn.reshape(ref_lat.shape[1:]) - ref_lat[0].numpy())
        print(f"vae_encoder:  max abs diff = {d.max():.3e}", flush=True)

        p_ncnn = run("vae_decoder", [ref_lat[0].numpy()])
        d = np.abs(p_ncnn.reshape(ref_pix.shape[1:]) - ref_pix[0].numpy())
        print(f"vae_decoder:  max abs diff = {d.max():.3e}", flush=True)

    print(f"done. models in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
