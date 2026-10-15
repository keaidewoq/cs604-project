"""Van: where does a 336 px training run lose time on this machine? Timing only, nothing trains.

Three measurements, each for every ``--configs`` entry:

1. Loader alone (CPU): the trainer's training transform over a fixed slice of the
   training set, for each ``--workers`` count, with and without colour jitter.
2. GPU step alone (synthetic images already on the GPU): forward, backward, clip,
   optimizer step, at each micro-batch x accumulation in ``--batches``, plain /
   fused AdamW / fused AdamW + torch.compile.
3. Both together, real loader into the real step: the config as it is (its workers,
   batch, plain AdamW), then with ``--e2e-workers`` workers, fused AdamW and compile.

If (3) as-is matches the slower of (1) at the config's workers and (2) plain, that one
is the limit. Prints a table and writes JSON to ``--out`` (default: next to the data).

    python scripts/van_speed_check.py --data_root /content/data \
        --configs configs/van_full_ft_336_fast.yaml configs/van_adaptformer_bias_final.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ImageListDataset, build_transforms  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.model import build_model  # noqa: E402
from src.train import adamw_param_groups, amp_context  # noqa: E402
from src.utils import load_config  # noqa: E402


def train_transform(cfg, jitter: bool):
    if jitter:
        return build_transforms(cfg)[0]
    from timm.data import create_transform, resolve_data_config
    from timm.models import get_pretrained_cfg

    pretrained = get_pretrained_cfg(cfg.model_name)
    pretrained = pretrained.to_dict() if hasattr(pretrained, "to_dict") else pretrained
    data_config = resolve_data_config({"input_size": (3, cfg.img_size, cfg.img_size)}, pretrained_cfg=pretrained)
    return create_transform(**data_config, is_training=True, color_jitter=None)


def image_slice(cfg, data_root, count, transform):
    dataset = ImageListDataset(data_root, "train", transform=transform)
    stride = max(len(dataset) // count, 1)  # spread over all classes
    return Subset(dataset, range(0, stride * count, stride))


def loader_rate(dataset, batch, workers, warm_batches):
    # The first `workers` batches arrive together once the workers start; skip twice that.
    loader = DataLoader(dataset, batch_size=batch, num_workers=workers, pin_memory=True, drop_last=True)
    seen, started = 0, None
    for index, (images, _labels, _paths) in enumerate(loader):
        if index == warm_batches:
            started = time.perf_counter()
        elif index > warm_batches:
            seen += len(images)
    return seen / (time.perf_counter() - started)


def make_step(cfg, device, fused, compiled):
    torch._dynamo.reset()
    model = build_model(cfg).to(device).train()
    loss_fn = build_loss(cfg, model)
    groups = adamw_param_groups(model, float(cfg.weight_decay), float(cfg.lr),
                                head_lr=cfg.get("head_lr"), layer_decay=cfg.get("layer_decay"))
    optimizer = AdamW(groups, lr=float(cfg.lr), **({"fused": True} if fused else {}))
    forward = torch.compile(model) if compiled else model
    params = [p for p in model.parameters() if p.requires_grad]
    clip = cfg.get("clip_grad_norm")

    def step(micro_batches):
        for images, labels in micro_batches:
            with amp_context(device, True):
                loss = loss_fn(forward(images), labels)
            (loss / len(micro_batches)).backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(params, float(clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss
    return model, step


def gpu_rate(cfg, device, micro, accum, fused, compiled, warm, steps):
    model, step = make_step(cfg, device, fused, compiled)
    images = torch.randn(micro, 3, cfg.img_size, cfg.img_size, device=device)
    labels = torch.randint(0, int(cfg.num_classes), (micro,), device=device)
    batches = [(images, labels)] * accum
    torch.cuda.reset_peak_memory_stats(device)
    for index in range(warm + steps):
        if index == warm:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
        loss = step(batches)
        loss.item()  # the trainer syncs every micro-batch for its log
    torch.cuda.synchronize(device)
    rate = steps * micro * accum / (time.perf_counter() - started)
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    del model, step
    torch.cuda.empty_cache()
    return rate, peak


def e2e_rate(cfg, device, dataset, workers, fused, compiled, warm, steps):
    micro, accum = int(cfg.batch_size), max(int(cfg.grad_accum_steps), 1)
    model, step = make_step(cfg, device, fused, compiled)
    loader = DataLoader(dataset, batch_size=micro, num_workers=workers, pin_memory=True,
                        drop_last=True, shuffle=True, persistent_workers=workers > 0)
    seen, done, pending, started = 0, 0, [], None
    while done < warm + steps:
        for images, labels, _paths in loader:
            pending.append((images.to(device, non_blocking=True), labels.to(device, non_blocking=True)))
            if len(pending) < accum:
                continue
            step(pending).item()
            pending = []
            done += 1
            if done == warm:
                torch.cuda.synchronize(device)
                started = time.perf_counter()
            elif done > warm:
                seen += micro * accum
            if done >= warm + steps:
                break
    torch.cuda.synchronize(device)
    rate = seen / (time.perf_counter() - started)
    del model, step, loader
    torch.cuda.empty_cache()
    return rate


def guarded(fn, *args):
    try:
        return fn(*args)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "OOM"
    except Exception as error:  # e.g. compile failing on this machine: report, keep going
        torch.cuda.empty_cache()
        return f"error: {type(error).__name__}: {str(error).splitlines()[0][:120]}"


def fmt(value):
    if isinstance(value, tuple):
        return f"{value[0]:7.0f} img/s  peak {value[1]:.1f} GB"
    return f"{value:7.0f} img/s" if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=None, help="default: 8, 10, 12 (capped at CPUs)")
    parser.add_argument("--batches", nargs="+", default=["128x2", "256x1"], help="micro-batch x accumulation")
    parser.add_argument("--images", type=int, default=7680,
                        help="images per loader measurement (the first 2 x workers batches are warm-up)")
    parser.add_argument("--repeats", type=int, default=2, help="loader timings per setting (best kept)")
    parser.add_argument("--warm", type=int, default=8, help="warm-up optimizer steps (compile happens here)")
    parser.add_argument("--steps", type=int, default=20, help="timed optimizer steps")
    parser.add_argument("--e2e-workers", type=int, default=None, help="default: the largest --workers")
    parser.add_argument("--skip", nargs="*", default=[], choices=["loader", "gpu", "e2e"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    cpus = os.cpu_count() or 1
    workers = args.workers or sorted({w for w in (8, 10, 12) if w <= cpus} or {cpus})
    e2e_workers = args.e2e_workers or max(workers)
    report = {"gpu": torch.cuda.get_device_name(device), "cpus": cpus, "torch": torch.__version__,
              "settings": vars(args), "results": {}}
    print(f"[speed] {report['gpu']}, {cpus} CPUs, torch {torch.__version__}", flush=True)

    def record(config_name, key, value):
        report["results"].setdefault(config_name, {})[key] = value
        print(f"[speed] {config_name:32s} {key:44s} {fmt(value)}", flush=True)

    for config_path in args.configs:
        cfg = load_config(config_path)
        name = Path(config_path).stem
        print(f"[speed] {name}: method={cfg.method} img_size={cfg.img_size} "
              f"batch={cfg.batch_size}x{cfg.grad_accum_steps} workers={cfg.num_workers}", flush=True)
        if "loader" not in args.skip:
            for jitter in (True, False):
                dataset = image_slice(cfg, args.data_root, args.images, train_transform(cfg, jitter))
                loader_rate(dataset, 128, max(workers), 2)  # first pass reads the files into the page cache
                for count in workers:
                    # Single timings are noisy (a slow worker start); keep the best of --repeats.
                    rates = [guarded(loader_rate, dataset, 128, count, 2 * count) for _ in range(args.repeats)]
                    numbers = [r for r in rates if isinstance(r, float)]
                    record(name, f"loader {count} workers, jitter {'on' if jitter else 'off'}",
                           max(numbers) if numbers else rates[0])
        if "gpu" not in args.skip:
            for spec in args.batches:
                micro, accum = (int(x) for x in spec.lower().split("x"))
                for label, fused, compiled in (("plain", False, False), ("fused", True, False),
                                               ("fused+compile", True, True)):
                    record(name, f"gpu {micro}x{accum} {label}",
                           guarded(gpu_rate, cfg, device, micro, accum, fused, compiled, args.warm, args.steps))
        if "e2e" not in args.skip:
            dataset = image_slice(cfg, args.data_root, args.images * 2, build_transforms(cfg)[0])
            record(name, f"e2e as configured ({cfg.num_workers} workers, plain)",
                   guarded(e2e_rate, cfg, device, dataset, int(cfg.num_workers), False, False, args.warm, args.steps))
            record(name, f"e2e {e2e_workers} workers, fused+compile",
                   guarded(e2e_rate, cfg, device, dataset, e2e_workers, True, True, args.warm, args.steps))

    out = Path(args.out) if args.out else Path(args.data_root) / f"van_speed_check_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[speed] wrote {out}")


if __name__ == "__main__":
    main()
