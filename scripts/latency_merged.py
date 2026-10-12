"""Inference latency of a trained LoRA run, unmerged vs merged (no retraining).

Loads <run_dir>/config.yaml and <run_dir>/last.pt, checks that merging LoRA into
the base weights leaves the predictions unchanged, then measures forward latency
in one session for: the model as trained (unmerged), the merged model, and a
plain ViT-B with the same head and no LoRA (reference). Same protocol as the
launchers: synthetic inputs, bf16, 30 warm-ups + 200 timed iterations, batch 1 and 64.

Writes <run_dir>/inference_latency_merged.json.

  python scripts/latency_merged.py --run_dir /content/drive/MyDrive/CS604_RUNS/lora_20261007_051926
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_latency import measure
from src.model import build_model
from src.peft.lora import merge_lora
from src.predict import load_checkpoint
from src.utils import apply_overrides, get_device, load_config, set_seed


def quiet_build(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return build_model(cfg)


@torch.no_grad()
def logits_on(model, inputs, device, use_bf16):
    out = []
    for x in inputs:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out.append(model(x.to(device)).float().cpu())
    return torch.cat(out)


def latency(model, img_size, device, use_bf16, warmup, iters):
    rows = []
    for bs in (1, 64):
        batch = torch.randn(bs, 3, img_size, img_size, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            ms, _ = measure(model, batch, device, warmup, iters)
        rows.append({"batch_size": bs, "ms_per_image": ms, "ms_per_batch": ms * bs})
        del batch
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True, help="Run folder with config.yaml and last.pt")
    p.add_argument("--ckpt", default="last.pt")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--check_images", type=int, default=256, help="Synthetic images for the equivalence check")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    if cfg.method != "lora":
        raise SystemExit(f"{run_dir} is a {cfg.method!r} run; merging only applies to LoRA")
    apply_overrides(cfg, pretrained=True)  # frozen weights come from the pretrained model
    set_seed(int(cfg.seed))
    device = get_device()
    use_bf16 = bool(cfg.bf16) and device.type == "cuda"
    img = int(cfg.img_size)

    model = quiet_build(cfg).to(device)
    with contextlib.redirect_stdout(io.StringIO()):
        load_checkpoint(model, run_dir / args.ckpt, device)
    model.eval()

    # 1. Equivalence check on the same inputs, before and after merging
    gen = torch.Generator().manual_seed(0)
    inputs = [torch.randn(32, 3, img, img, generator=gen) for _ in range(max(args.check_images // 32, 1))]
    before = {m: logits_on(model, inputs, device, m) for m in (False, use_bf16)}
    unmerged = latency(model, img, device, use_bf16, args.warmup, args.iters)

    merge_lora(model)
    left = sum(1 for m in model.modules() if type(m).__name__ == "LoRALinear")
    after = {m: logits_on(model, inputs, device, m) for m in (False, use_bf16)}
    check = {
        "images": sum(x.size(0) for x in inputs),
        "lora_layers_left_after_merge": left,
        "fp32_max_abs_logit_diff": float((before[False] - after[False]).abs().max()),
        "fp32_top1_agreement": float((before[False].argmax(1) == after[False].argmax(1)).float().mean()),
    }
    if use_bf16:
        check["bf16_top1_agreement"] = float((before[True].argmax(1) == after[True].argmax(1)).float().mean())
    merged = latency(model, img, device, use_bf16, args.warmup, args.iters)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Reference: the same ViT-B and head with no LoRA at all
    ref_cfg = load_config(run_dir / "config.yaml")
    apply_overrides(ref_cfg, pretrained=False, method="linear_probe")
    ref = quiet_build(ref_cfg).to(device).eval()
    plain = latency(ref, img, device, use_bf16, args.warmup, args.iters)

    results = {
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / args.ckpt),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": str(torch.__version__),
        "precision": "bf16" if use_bf16 else "fp32",
        "img_size": img,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iters,
        "scope": "Forward pass on synthetic inputs; excludes loading, preprocessing and transfers",
        "equivalence_check": check,
        "unmerged": unmerged,
        "merged": merged,
        "plain_vit_same_head": plain,
    }
    out = run_dir / "inference_latency_merged.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"[merge] check: {json.dumps(check)}")
    for name in ("unmerged", "merged", "plain_vit_same_head"):
        bs1, bs64 = results[name]
        print(f"[latency] {name:20s} bs1 {bs1['ms_per_image']:.2f} ms/image | bs64 {bs64['ms_per_image']:.3f} ms/image")
    print(f"[latency] wrote {out}")


if __name__ == "__main__":
    main()
