# vosr-ncnn

ncnn deployment of [VOSR](https://github.com/cswry/VOSR) (VOSR2, 1.4B one-step DiT,
generative super-resolution). Runs on CPU and Vulkan GPU.
The conversion approach follows [pixRestore-ncnn](https://github.com/lll143653/pixRestore-ncnn).

Single-tile fixed shape: LR image ×4 → 512×512 (latent 64×64, token grid 32×32).
Images of any size are handled by tiling (Python demo).

## Model split (5 ncnn models)

| model | input | output | notes |
|---|---|---|---|
| `vae_encoder` | (1,3,512,512) pixels [-1,1] | (1,16,64,64) | Qwen-Image 2D VAE encode + posterior.mode + latent normalization (baked in) |
| `dinov2l_feat` | (1,3,448,448) ImageNet-normalized | (1,1024,1024) | DINOv2-L layer-17 patch tokens (cls dropped) |
| `vosr_dit_a` | lq_latent + noise + z_raw | (1,1024,1536) | DiT patch embed + blocks 0..17 |
| `vosr_dit_b` | x_tokens + z_raw | (1,1024,64) | DiT blocks 18..35 + final layer, velocity patch tokens |
| `vae_decoder` | (1,16,64,64) | (1,3,512,512) [-1,1] | latent denormalization + decode + clamp (baked in) |

The DiT is split into two chained halves because pnnx's `pass_ncnn` crashes on the
full 5.5 GB fp32 graph; the two ~2.7 GB halves convert fine. At runtime the output
tokens of `vosr_dit_a` feed directly into `vosr_dit_b`.

Host-side glue: unpatchify (2×2×16 → spatial), `sr_latent = noise - u` (one-step
Euler, t: 1 → 0), bicubic upsampling, ImageNet normalization, wavelet color fix
(σ=5 Gaussian low/high split against the upscaled input), noise generation.

Key conversion tricks (all verified max abs diff == 0 on the torch side):

- one-step sampling: t = 1.0 constant and VOSR2 has no r_embedder, so the
  t_embedder/t_block output and every block's AdaLN modulation table are
  precomputed once and baked in as buffers.
- interleaved RoPE `rotate_half` is rewritten as a constant permutation-matrix
  multiplication (5D reshapes break pnnx → ncnn conversion).
- SDPA is decomposed into matmul + softmax at trace time (ncnn has no portable
  SDPA layer across versions).
- unpatchify and other 6D reshapes stay on the host.

## 1. Clone VOSR and set up the environment

```bash
git clone https://github.com/cswry/VOSR.git
conda create -n vosr python=3.10 -y
conda activate vosr

# conversion deps only (VOSR's training requirements like basicsr are not needed)
pip install torch==2.5.1 timm diffusers einops safetensors fairscale pillow
pip install pnnx ncnn==1.0.20260526
```

## 2. Download the weights

```bash
python scripts/download_weights.py --output <VOSR repo>/preset/ckpts
```

Fetches ~6.5 GB from ModelScope ([LULALULALU/VOSR_CKPT](https://modelscope.cn/models/LULALULALU/VOSR_CKPT)):

```
preset/ckpts/
├── VOSR2/                                       # 1.4B one-step DiT checkpoint + args.json
├── Qwen-Image-vae-2d/                           # Qwen 2D VAE (config + weights)
└── torch_cache/
    ├── checkpoints/dinov2_vitl14_pretrain.pth   # DINOv2-L weights
    └── facebookresearch_dinov2_main/            # DINOv2 hub source tree (offline torch.hub)
```

Existing files verified by size are skipped and partial downloads resume, so it is
safe to re-run after an interruption.

## 3. Convert the models

```bash
python scripts/convert.py \
    --repo <VOSR repo> \
    --checkpoint <VOSR repo>/preset/ckpts/VOSR2 \
    --output models
```

One command does: checkpoint loading → torch parity check (deploy graphs vs stock
modules) → trace → pnnx → ncnn smoke validation. Output (fp32):

```
models/
├── vae_encoder.ncnn.param / .bin     #   77 MB
├── dinov2l_feat.ncnn.param / .bin    #  914 MB
├── vosr_dit_a.ncnn.param / .bin      # 2.77 GB
├── vosr_dit_b.ncnn.param / .bin      # 2.77 GB
└── vae_decoder.ncnn.param / .bin     #  101 MB
```

fp16 storage (half the size, ncnn tool, flag 65536):

```bash
for m in vae_encoder dinov2l_feat vosr_dit_a vosr_dit_b vae_decoder; do
  ncnnoptimize models/$m.ncnn.param models/$m.ncnn.bin \
               models/${m}_fp16.ncnn.param models/${m}_fp16.ncnn.bin 65536
done
# full set: 6.5 GB -> 3.3 GB
```

Note on pnnx: the pip build of pnnx may crash inside `pass_ncnn` on the ~2.7 GB DiT
halves on some platforms. If that happens, build pnnx from the ncnn source tree
(`tools/pnnx`) and pass it explicitly: `--pnnx-exe <ncnn>/tools/pnnx/build/src/pnnx`
(the pnnx binary needs its matching libtorch shared libraries on the library path).

## 4. Python demo

```bash
python scripts/run_inference.py -m models -i lr_128x128.png -o out.png -u 4
# fp16 storage models: point -m at a dir containing the *_fp16.ncnn.* files
# Vulkan acceleration: add --gpu (fp32 compute is forced internally)
```

Inputs larger than 128×128 automatically switch to tiled mode: the LR image is
processed in 128×128 tiles with stride 96 (32 px overlap), merged with linear-ramp
blending, and the wavelet color fix runs once on the merged image.

IMPORTANT: always feed ncnn with ncnn-allocated aligned Mats and copy data in
(`ncnn.Mat(numpy)` is zero-copy and numpy buffers are not guaranteed 32-byte
aligned → SIMD kernels can segfault). `run_inference.py` already does this.

Alignment results (128×128 RealSR test image, shared noise):

- C++ fp32 vs Python ncnn full pipeline: PSNR **72.1 dB**
- C++/Python fp32 vs torch GPU (final PNG, after color fix): PSNR **51.1 dB**
- fp16 storage vs fp32 (C++): PSNR **63.1 dB**; fp16 vs torch GPU also **51.1 dB**

## 5. C++ demo

ncnn SDK: build from <https://github.com/Tencent/ncnn> (Vulkan recommended) or use a release package.

OpenCV-mobile: download a prebuilt release from
<https://github.com/nihui/opencv-mobile/releases> and place it under `3rdparty/`.

```bash
cmake -S cpp -B cpp/build \
    -DNCNN_INCLUDE_DIR=<ncnn install>/include/ncnn \
    -DNCNN_LIB_DIR=<ncnn install>/lib
cmake --build cpp/build --config Release -j

# Vulkan (default when a Vulkan device is found; fp32 compute is forced to keep norm precision)
./cpp/build/vosr_sr models lr_128x128.png out.png --fp16
# CPU fp32
./cpp/build/vosr_sr models lr_128x128.png out.png --cpu
# CPU fp16 (lower memory)
./cpp/build/vosr_sr models lr_128x128.png out.png --cpu --fp16
# reproduce a torch run exactly: --noise noise.bin (16*64*64 fp32 CHW); --dump-out exports decoded float pixels
```

Measured latency (RTX 3090 + desktop CPU, 8 threads, 128×128 → 512×512):

| backend | vae_encoder | dinov2 | DiT (a+b) | vae_decoder | total |
|---|---|---|---|---|---|
| CPU fp32 | 1.0 s | 1.0 s | 5.8 s | 1.8 s | ≈ **9.7 s / image** |
| Vulkan fp16-storage | 0.19 s | 0.56 s | 3.0 s | 0.13 s | ≈ **3.9 s / image** |

## Known limitations / future work

- The C++ demo currently runs a single fixed 512×512 tile (input must be 128×128,
  or ×4 of it must be exactly 512×512). The Python demo supports arbitrary sizes
  via tiling (see above). Arbitrary sizes in C++ need the same tile framework
  (reference: pixrestore.cpp in pixRestore-ncnn).
- Vulkan forces fp32 compute (fp16 arithmetic hurts RMSNorm/LayerNorm precision);
  the process ends with `_exit(0)` to skip ncnn/Vulkan static destruction, which
  segfaults with some driver combinations.
- The multi-step model (VOSR_1.4B_ms) and the 0.5B variant (SD2.1 VAE) are not
  converted; same flow — re-run `convert.py` with the corresponding config.
- INT8 quantization is not done: diffusion-style models are risky to quantize and
  the DiT body is Linear/MatMul, so it needs calibrated quantization
  (ncnn2table/ncnn2int8) with per-layer fallback.
