"""Van: inference latency of a trained run with the zoom view (two passes), no retraining.

Loads <run_dir>/config.yaml and <run_dir>/last.pt and times, in one session:

  single      one forward pass (as scripts/benchmark_latency.py)
  zoom        the submission path of ``src.predict --zoom`` (``_predict_zoom``): plain pass,
              class-activation box, crop from the 2x copy, second pass, softmax average

With --merge_lora, every LoRALinear is folded into its base weight (W + B A * alpha/rank)
after an fp32 equivalence check, and both are timed again ("merged_single", "merged_zoom").
This works for Van's AdaptFormer + LoRA runs; the AdaptFormer branch itself cannot be merged.

Same protocol as the team's latency files: synthetic inputs already on the GPU, bf16,
30 warm-ups + 200 timed iterations, batch 1 and 64. Excludes image loading, the initial
resize/crop and CPU-to-GPU transfers. Writes <run_dir>/inference_latency_zoom.json
(or --out).

  python scripts/van_latency_zoom.py --run_dir /content/drive/MyDrive/CS604_RUNS/full_ft_20261017_122237
  python scripts/van_latency_zoom.py --run_dir .../van_adaptformer_20261021_083823 --merge_lora
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import van_zoom
from src.model import build_model
from src.peft.lora import LoRALinear
from src.predict import _predict_zoom, load_checkpoint
from src.utils import apply_overrides, get_device, load_config, set_seed


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def time_fn(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - start) / iters * 1000.0


def latency_rows(model, img, device, use_bf16, zoom_mass, batch_sizes, warmup, iters):
    """{"single": [...], "zoom": [...]} with ms per batch and per image for each batch size."""
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    out = {"single": [], "zoom": []}
    gen = torch.Generator(device=device).manual_seed(0)
    for bs in batch_sizes:
        x = torch.randn(bs, 3, img, img, device=device, generator=gen)
        big = torch.randn(bs, 3, img * van_zoom.FACTOR, img * van_zoom.FACTOR, device=device, generator=gen)

        def single():
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
                model(x)

        def zoom():
            _predict_zoom(model, (x, big), device, amp_dtype, use_bf16, False, zoom_mass)

        for name, fn in (("single", single), ("zoom", zoom)):
            ms = time_fn(fn, device, warmup, iters)
            out[name].append({"batch_size": bs, "ms_per_batch": ms, "ms_per_image": ms / bs})
        del x, big
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def merge_lora_layers(model):
    """Replace every LoRALinear by its base Linear with the low-rank update added in."""
    merged = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                base = child.linear
                delta = child.lora_B.weight @ child.lora_A.weight * child.scaling
                with torch.no_grad():
                    base.weight.add_(delta.to(base.weight.dtype))
                setattr(module, name, base)
                merged += 1
    return merged


@torch.no_grad()
def fp32_logits(model, inputs):
    return torch.cat([model(x).float() for x in inputs])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run_dir", required=True, help="Run folder with config.yaml and the checkpoint")
    p.add_argument("--ckpt", default="last.pt")
    p.add_argument("--zoom_mass", type=float, default=van_zoom.DEFAULT_MASS)
    p.add_argument("--merge_lora", action="store_true", help="Also time with LoRA merged into the weights")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 64])
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--out", default=None, help="Output JSON (default <run_dir>/inference_latency_zoom.json)")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / args.ckpt
    cfg = load_config(run_dir / "config.yaml")
    info = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    partial = bool(info.get("trainable_only", False)) if isinstance(info, dict) else False
    del info
    # A compact (trainable-only) checkpoint needs the frozen weights from the pretrained model.
    apply_overrides(cfg, pretrained=partial)
    set_seed(int(cfg.seed))
    device = get_device()
    use_bf16 = bool(cfg.bf16) and device.type == "cuda"
    img = int(cfg.img_size)

    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(cfg).to(device)
        load_checkpoint(model, ckpt_path, device)
    model.eval()
    lora_layers = sum(isinstance(m, LoRALinear) for m in model.modules())
    print(f"[latency] {run_dir.name}: method={cfg.method} img_size={img} bf16={use_bf16} "
          f"lora_layers={lora_layers} device={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

    results = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "method": cfg.method,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "gpu_capacity_GiB": torch.cuda.get_device_properties(0).total_memory / 1024**3 if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "precision": "bf16" if use_bf16 else "fp32",
        "img_size": img,
        "zoom_mass": args.zoom_mass,
        "tta_flip": False,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iters,
        "scope": ("Synthetic views already on GPU. 'single' = one forward pass; 'zoom' = src.predict's "
                  "_predict_zoom (plain pass, box selection, crop from the 2x copy, second pass, softmax "
                  "average). Excludes image loading, initial resize/crop and CPU-to-GPU transfers."),
        "lora_layers": lora_layers,
    }
    results.update(latency_rows(model, img, device, use_bf16, args.zoom_mass,
                                args.batch_sizes, args.warmup, args.iters))

    if args.merge_lora:
        if not lora_layers:
            raise SystemExit(f"{run_dir} has no LoRA layers to merge")
        gen = torch.Generator(device=device).manual_seed(1)
        inputs = [torch.randn(32, 3, img, img, device=device, generator=gen) for _ in range(4)]
        before = fp32_logits(model, inputs)
        merged = merge_lora_layers(model)
        after = fp32_logits(model, inputs)
        results["merge_check"] = {
            "lora_layers_merged": merged,
            "lora_layers_left": sum(isinstance(m, LoRALinear) for m in model.modules()),
            "images": sum(x.size(0) for x in inputs),
            "fp32_max_abs_logit_diff": float((before - after).abs().max()),
            "fp32_top1_agreement": float((before.argmax(1) == after.argmax(1)).float().mean()),
        }
        print(f"[latency] merge check: {json.dumps(results['merge_check'])}")
        if results["merge_check"]["fp32_top1_agreement"] < 0.99:
            raise SystemExit("merged model disagrees with the unmerged one; not timing it")
        merged_rows = latency_rows(model, img, device, use_bf16, args.zoom_mass,
                                   args.batch_sizes, args.warmup, args.iters)
        results["merged_single"] = merged_rows["single"]
        results["merged_zoom"] = merged_rows["zoom"]

    out = Path(args.out) if args.out else run_dir / "inference_latency_zoom.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name in ("single", "zoom", "merged_single", "merged_zoom"):
        if name in results:
            rows = " | ".join(f"bs{r['batch_size']} {r['ms_per_image']:.3f} ms/image" for r in results[name])
            print(f"[latency] {name:14s} {rows}")
    print(f"[latency] wrote {out}")


if __name__ == "__main__":
    main()
