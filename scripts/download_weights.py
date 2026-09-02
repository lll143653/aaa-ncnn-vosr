#!/usr/bin/env python
"""Download the VOSR2 inference files from ModelScope (LULALULALU/VOSR_CKPT).

Fetches only what VOSR2 one-step inference needs, into the VOSR repo's preset/ckpts layout:

  <output>/VOSR2/                                        # 1.4B one-step DiT checkpoint + args.json
  <output>/Qwen-Image-vae-2d/                          # Qwen 2D VAE (config + weights)
  <output>/torch_cache/checkpoints/dinov2_vitl14_pretrain.pth    # DINOv2-L weights
  <output>/torch_cache/facebookresearch_dinov2_main/           # DINOv2 hub source tree (offline torch.hub)

Usage:
  python scripts/download_weights.py --output /path/to/VOSR-main/preset/ckpts

Stdlib only. Existing files verified by size are skipped; partial downloads resume via Range requests;
safe to re-run after an interruption. Other variants (multi-step, 0.5B), training-metric ckpts
and the SD2.1 VAE are intentionally not downloaded.
"""
import argparse
import concurrent.futures
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://modelscope.cn/api/v1/models/LULALULALU/VOSR_CKPT/repo/files?Recursive=true"
RESOLVE = "https://modelscope.cn/models/LULALULALU/VOSR_CKPT/resolve/master/"

ALLOW_PREFIXES = ("VOSR2/", "Qwen-Image-vae-2d/", "torch_cache/facebookresearch_dinov2_main/")
DINO_PTH = "torch_cache/checkpoints/dinov2_vitl14_pretrain.pth"


def needed_files():
    """Return [(repo_path, size_bytes), ...] for every file we need."""
    with urllib.request.urlopen(API) as r:
        files = json.load(r)["Data"]["Files"]
    keep = []
    for f in files:
        if f["Type"] != "blob":
            continue
        path = f["Path"]
        if path == DINO_PTH or path.startswith(ALLOW_PREFIXES):
            keep.append((path, f["Size"]))
    return keep


def download_one(path, size, out_dir, tries=8):
    """Download one file to out_dir/path, resuming partial downloads. Returns an error string or None."""
    dst = out_dir / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size == size:
        return None
    tmp = dst.with_name(dst.name + ".part")
    url = RESOLVE + urllib.parse.quote(path)
    last_exc = None
    for attempt in range(tries):
        try:
            got = tmp.stat().st_size if tmp.is_file() else 0
            req = urllib.request.Request(url)
            if got:
                req.add_header("Range", "bytes=%d-" % got)
            with urllib.request.urlopen(req) as r, open(tmp, "ab") as w:
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    w.write(chunk)
            if tmp.stat().st_size == size:
                tmp.replace(dst)
                return None
            # short read; next attempt resumes via Range
        except Exception as e:
            last_exc = e
    if last_exc is not None:
        return "%s: %s: %s" % (path, type(last_exc).__name__, last_exc)
    return "%s: size mismatch after %d attempts" % (path, tries)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True,
                    help="target dir: preset/ckpts of the VOSR repo (VOSR2/, Qwen-Image-vae-2d/, torch_cache/ are written inside it)")
    args = ap.parse_args()
    out_dir = Path(args.output).resolve()

    files = needed_files()
    total = sum(s for _, s in files)
    print("%d files, %.2f GB total -> %s" % (len(files), total / 2 ** 30, out_dir), flush=True)

    def job(item):
        path, size = item
        err = download_one(path, size, out_dir)
        if not err:
            print("  ok  %s (%.0f MB)" % (path, size / 2 ** 20), flush=True)
        return err

    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        fails = [e for e in ex.map(job, files) if e]
    if fails:
        print("\n".join(fails))
        sys.exit("some downloads failed; re-run the same command to resume")
    print("done")


if __name__ == "__main__":
    main()
