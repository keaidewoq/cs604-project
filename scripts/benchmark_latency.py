"""Measure inference latency (ms/image) at batch size 1 and 64."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import build_model
from src.utils import apply_overrides, get_device, load_config, set_seed


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure(model, batch, device, warmup, iters):
    model.eval()
    for _ in range(warmup):
        _ = model(batch)
    sync(device)

    start = time.perf_counter()
    for _ in range(iters):
        _ = model(batch)
    sync(device)
    elapsed = time.perf_counter() - start
    ms_per_image = (elapsed / iters / batch.size(0)) * 1000.0
    return ms_per_image, elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark forward-pass latency.")
    parser.add_argument("--config", default="configs/linear_probe.yaml")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--pretrained", action="store_true", help="Load pretrained weights (off by default).")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    apply_overrides(cfg, pretrained=args.pretrained)
    set_seed(int(cfg.seed))
    device = get_device()

    model = build_model(cfg).to(device).eval()
    print(f"[bench] device={device} method={cfg.method} img_size={cfg.img_size}")

    use_bf16 = bool(cfg.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    for bs in (1, 64):
        batch = torch.randn(bs, 3, cfg.img_size, cfg.img_size, device=device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
            ms, elapsed = measure(model, batch, device, args.warmup, args.iters)
        print(
            f"[bench] bs={bs:2d}  {ms:.3f} ms/image  "
            f"({elapsed * 1000.0 / args.iters:.2f} ms/batch, {args.iters} timed iters)"
        )


if __name__ == "__main__":
    main()
