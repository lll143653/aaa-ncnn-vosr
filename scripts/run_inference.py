#!/usr/bin/env python
"""End-to-end VOSR2 inference on the four ncnn models (CPU).

Pipeline (mirrors inference_vosr_onestep.py):
  LR image --PIL bicubic--> 512x512 [-1,1]
    -> vae_encoder  -> lq_latent (16,64,64)
    -> torch-bicubic resize to 448 + ImageNet norm -> dinov2l_feat -> z_raw
    -> vosr_dit(lq_latent, noise, z_raw) -> tokens
    -> host unpatchify -> sr_latent = noise - u   (one-step Euler, t: 1 -> 0)
    -> vae_decoder -> pixels [-1,1]
    -> wavelet color fix against the bicubic-upscaled input

IMPORTANT: always feed ncnn with ncnn-allocated aligned Mats and copy data in.
ncnn.Mat(numpy) is zero-copy and numpy buffers are not guaranteed 32-byte
aligned -> SIMD kernels can segfault.

Usage:
  python scripts/run_inference.py -m models -i lr_128x128.png -o out.png -u 4
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

RES = 512          # fixed pixel tile
LAT = 64           # latent side
GRID = 32          # token grid side
PATCH = 2
ZC = 16            # latent channels
DINO = 448
ENC_DIM = 1024

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# torch-compatible bicubic resize (antialias=False, align_corners=False)
# ---------------------------------------------------------------------------

def _bicubic_coeffs(in_size, out_size):
    a = -0.75
    scale = in_size / out_size
    pos = (np.arange(out_size, dtype=np.float64) + 0.5) * scale - 0.5
    base = np.floor(pos)
    frac = pos - base
    idx = base[:, None] + np.arange(-1, 3, dtype=np.float64)[None, :]  # (out,4)
    x = np.abs(idx - pos[:, None])
    w = np.where(x <= 1.0,
                 (a + 2) * x**3 - (a + 3) * x**2 + 1.0,
                 a * x**3 - 5 * a * x**2 + 8 * a * x - 4 * a)
    w = np.where(x >= 2.0, 0.0, w)
    idx = np.clip(idx, 0, in_size - 1).astype(np.int64)
    return idx, w.astype(np.float32)


def resize_bicubic_torch(img, out_h, out_w):
    """img (H,W,C) float32 -> (out_h,out_w,C), torch F.interpolate bicubic."""
    iy, wy = _bicubic_coeffs(img.shape[0], out_h)
    ix, wx = _bicubic_coeffs(img.shape[1], out_w)
    tmp = np.einsum('ok,okjc->ojc', wy, img[iy])         # vertical: (out_h, W, C)
    out = np.einsum('ok,jokc->joc', wx, tmp[:, ix])      # horizontal: (out_h, out_w, C)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# ncnn helpers
# ---------------------------------------------------------------------------

def to_mat(arr):
    """numpy (C,H,W) or (N,C) fp32 -> ncnn-allocated aligned Mat (copy in)."""
    import ncnn
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        m = ncnn.Mat(arr.shape[2], arr.shape[1], arr.shape[0])
    elif arr.ndim == 2:
        m = ncnn.Mat(arr.shape[1], arr.shape[0])
    else:
        raise ValueError(f'bad shape {arr.shape}')
    np.array(m, copy=False)[:] = arr
    return m


class NcnnNet:
    def __init__(self, model_dir, name, num_threads=8, use_gpu=False):
        import ncnn
        self.net = ncnn.Net()
        self.net.opt.num_threads = num_threads
        if use_gpu:
            self.net.opt.use_vulkan_compute = 1
            # Vulkan fp16 breaks RMSNorm/LayerNorm precision - force fp32 compute
            self.net.opt.use_fp16_packed = 0
            self.net.opt.use_fp16_storage = 0
            self.net.opt.use_fp16_arithmetic = 0
        assert self.net.load_param(os.path.join(model_dir, f'{name}.ncnn.param')) == 0
        assert self.net.load_model(os.path.join(model_dir, f'{name}.ncnn.bin')) == 0

    def __call__(self, *inputs):
        ex = self.net.create_extractor()
        keep = []
        for i, arr in enumerate(inputs):
            m = to_mat(arr)
            keep.append(m)
            ex.input(f'in{i}', m)
        ret, out = ex.extract('out0')
        assert ret == 0
        return np.array(out, dtype=np.float32, copy=True)


def unpatchify(tokens, grid=GRID, patch=PATCH, channels=ZC):
    """(grid*grid, patch*patch*C) -> (C, grid*patch, grid*patch)."""
    x = tokens.reshape(grid, grid, patch, patch, channels)
    x = x.transpose(4, 0, 2, 1, 3)      # c h p w q
    return x.reshape(channels, grid * patch, grid * patch)


def wavelet_color_fix(target01, source01, sigma=5):
    """Low frequency from source, high frequency from target. Both (H,W,3)."""
    import cv2
    src_low = cv2.GaussianBlur(source01, (0, 0), sigma)
    tgt_low = cv2.GaussianBlur(target01, (0, 0), sigma)
    return np.clip(src_low + target01 - tgt_low, 0, 1)


def sr_pre(nets, tile_img):
    """Pre-DiT stage: host resize + vae_encoder + dinov2 -> (lq_latent, z_raw)."""
    venc, dino = nets[0], nets[1]
    up = tile_img.resize((RES, RES), Image.BICUBIC)          # PIL 8bit bicubic
    up01 = np.asarray(up, dtype=np.float32) / 255.0
    lq = up01.transpose(2, 0, 1) * 2.0 - 1.0                 # (3,512,512) [-1,1]

    lq_latent = venc(lq)                                     # (16,64,64)

    dino_in = resize_bicubic_torch(up01, DINO, DINO).clip(0, 1)
    dino_in = (dino_in - IMAGENET_MEAN) / IMAGENET_STD
    z_raw = dino(dino_in.transpose(2, 0, 1))                 # (1024,1024)
    return lq_latent, z_raw


def sr_dit(nets, lq_latent, noise, z_raw):
    """DiT stage (the bottleneck): -> sr_latent (16,64,64)."""
    dit_a, dit_b = nets[2], nets[3]
    x_tokens = dit_a(lq_latent, noise, z_raw)                # (1024,1536)
    tokens = dit_b(x_tokens, z_raw)                          # (1024,64)
    u = unpatchify(tokens)                                   # (16,64,64)
    return noise - u                                         # one step, t: 1 -> 0


def sr_post(nets, sr_latent):
    """Post-DiT stage: vae_decoder -> ((512,512,3) float [0,1], raw pix [-1,1])."""
    vdec = nets[4]
    pix = vdec(sr_latent)                                    # (3,512,512) [-1,1]
    return (pix.transpose(1, 2, 0) + 1.0) * 0.5, pix


def sr_tile(nets, tile_img, noise):
    """One 128x128 LR tile (PIL Image) -> ((512,512,3) [0,1], raw pix [-1,1])."""
    lq_latent, z_raw = sr_pre(nets, tile_img)
    sr_latent = sr_dit(nets, lq_latent, noise, z_raw)
    return sr_post(nets, sr_latent)


# ---------------------------------------------------------------------------
# tiling (arbitrary LR sizes; tile = RES/upscale LR px, overlap + ramp blending)
# ---------------------------------------------------------------------------

TILE_STRIDE = 96           # LR px; tile 128 -> 32 px overlap
RAMP = (128 - TILE_STRIDE) * 4   # 128 SR px blend width


def tile_starts(dim, tile, stride):
    """Evenly spaced starts; last tile is pulled back to end exactly at dim."""
    if dim <= tile:
        return [0]
    starts = list(range(0, dim - tile + 1, stride))
    if starts[-1] != dim - tile:
        starts.append(dim - tile)
    return starts


def ramp_weights(size, ramp, has_left, has_right):
    """1D weights: 1 inside, linear 0->1 over `ramp` px at edges with a neighbor."""
    w = np.ones(size, dtype=np.float32)
    if has_left:
        w[:ramp] = np.minimum(w[:ramp], np.arange(ramp, dtype=np.float32) / ramp)
    if has_right:
        w[-ramp:] = np.minimum(w[-ramp:], np.arange(ramp - 1, -1, -1, dtype=np.float32) / ramp)
    return w


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-m', '--model-dir', required=True)
    p.add_argument('-i', '--input', required=True)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('-u', '--upscale', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--noise', default=None,
                   help='optional fp32 CHW (16*64*64) noise .bin for reproduction '
                        '(single-tile mode only)')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--gpu', action='store_true', help='use Vulkan (fp32 compute)')
    p.add_argument('--no-color-fix', action='store_true')
    p.add_argument('--dump-out', default=None,
                   help='optional path to dump decoded float pixels (fp32, [-1,1])')
    args = p.parse_args()
    for path in (args.output, args.dump_out):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    img = Image.open(args.input).convert('RGB')
    w, h = img.size
    tw, th = w * args.upscale, h * args.upscale
    if args.upscale <= 0 or RES % args.upscale:
        sys.exit(f'upscale must be a positive divisor of {RES}')
    lr_tile = RES // args.upscale                            # 128 for u=4

    nets = (
        NcnnNet(args.model_dir, 'vae_encoder', args.threads, args.gpu),
        NcnnNet(args.model_dir, 'dinov2l_feat', args.threads, args.gpu),
        NcnnNet(args.model_dir, 'vosr_dit_a', args.threads, args.gpu),
        NcnnNet(args.model_dir, 'vosr_dit_b', args.threads, args.gpu),
        NcnnNet(args.model_dir, 'vae_decoder', args.threads, args.gpu),
    )

    if (tw, th) == (RES, RES):
        # ---- single tile (original path) ----
        if args.noise:
            noise = np.fromfile(args.noise, dtype=np.float32).reshape(ZC, LAT, LAT)
        else:
            noise = np.random.RandomState(args.seed).standard_normal(
                (ZC, LAT, LAT)).astype(np.float32)
        out01, pix = sr_tile(nets, img, noise)
        if args.dump_out:
            pix.astype(np.float32).tofile(args.dump_out)
        up01 = np.asarray(img.resize((RES, RES), Image.BICUBIC),
                          dtype=np.float32) / 255.0
        if not args.no_color_fix:
            out01 = wavelet_color_fix(out01, up01)
        Image.fromarray((out01 * 255.0).round().astype(np.uint8)).save(args.output)
        print(f'saved {args.output}', flush=True)
        os._exit(0)

    # ---- tiled mode (pipelined: pre/dit/post of different tiles overlap) ----
    # pre  = host resize + vae_encoder + dinov2   (background thread)
    # dit  = vosr_dit_a + vosr_dit_b              (main thread, the bottleneck)
    # post = vae_decoder + blending               (background thread)
    # ncnn releases the GIL during extract and one Extractor per call is
    # thread-safe, so the three stages of different tiles run concurrently.
    # Tiles are processed in the same row-major order and with the same
    # per-tile noise seeds as the serial version -> bit-identical output.
    from concurrent.futures import ThreadPoolExecutor
    xs = tile_starts(w, lr_tile, TILE_STRIDE)
    ys = tile_starts(h, lr_tile, TILE_STRIDE)
    n_tiles = len(xs) * len(ys)
    print(f'tiled: input {w}x{h} -> {tw}x{th}, grid {len(xs)}x{len(ys)} '
          f'({n_tiles} tiles of {lr_tile}px, stride {TILE_STRIDE})', flush=True)

    # per-axis blend weights per tile position (SR pixel space)
    wx = [ramp_weights(RES, RAMP, x > 0, x + lr_tile < w) for x in xs]
    wy = [ramp_weights(RES, RAMP, y > 0, y + lr_tile < h) for y in ys]
    positions = [(iy, y, ix, x) for iy, y in enumerate(ys) for ix, x in enumerate(xs)]

    acc = np.zeros((th, tw, 3), dtype=np.float32)
    wsum = np.zeros((th, tw, 1), dtype=np.float32)

    def pre(idx):
        iy, y, ix, x = positions[idx]
        tile = img.crop((x, y, x + lr_tile, y + lr_tile))
        lq_latent, z_raw = sr_pre(nets, tile)
        w2 = (wy[iy][:, None] * wx[ix][None, :])[..., None]
        return lq_latent, z_raw, w2, y * args.upscale, x * args.upscale

    def post(sr_latent, w2, ys_sr, xs_sr):
        out01, _ = sr_post(nets, sr_latent)
        acc[ys_sr:ys_sr + RES, xs_sr:xs_sr + RES] += out01 * w2
        wsum[ys_sr:ys_sr + RES, xs_sr:xs_sr + RES] += w2

    import time
    t0 = time.time()
    pre_pool = ThreadPoolExecutor(1)
    post_pool = ThreadPoolExecutor(1)
    pre_fut = pre_pool.submit(pre, 0)
    posts = []
    for idx in range(n_tiles):
        lq_latent, z_raw, w2, ys_sr, xs_sr = pre_fut.result()
        if idx + 1 < n_tiles:
            pre_fut = pre_pool.submit(pre, idx + 1)   # prefetch next tile's pre-stage
        noise = np.random.RandomState(args.seed + idx).standard_normal(
            (ZC, LAT, LAT)).astype(np.float32)
        sr_latent = sr_dit(nets, lq_latent, noise, z_raw)
        posts.append(post_pool.submit(post, sr_latent, w2, ys_sr, xs_sr))
        print(f'  tile {idx + 1}/{n_tiles} ({time.time() - t0:.1f}s elapsed)', flush=True)
    for f in posts:
        f.result()
    pre_pool.shutdown()
    post_pool.shutdown()

    out01 = acc / np.maximum(wsum, 1e-6)
    if not args.no_color_fix:
        print('wavelet color fix...', flush=True)
        up01 = np.asarray(img.resize((tw, th), Image.BICUBIC), dtype=np.float32) / 255.0
        out01 = wavelet_color_fix(out01, up01)
    Image.fromarray((out01 * 255.0).round().astype(np.uint8)).save(args.output)
    print(f'saved {args.output} ({time.time() - t0:.1f}s total)', flush=True)
    # ncnn-python's Vulkan teardown is buggy even on CPU-only runs in some
    # builds; hard-exit to avoid a shutdown crash.
    os._exit(0)


if __name__ == '__main__':
    main()
