"""Pilot sweep: short runs on a class subset to pick parameters before a full run.

python scripts/pilot_sweep.py --config configs/lora_r16_336px_qkv-proj-fc_lr-sweep.yaml \
    --data_root /content/data --output_dir /content/drive/MyDrive/CS604_RUNS

Reads the `pilot:` block of the method config:

    pilot:
      num_classes_subset: 1000   # keep ALL images of this many classes (~45/class)
      holdout_per_class: 5       # labelled holdout taken from those classes
      tie_tolerance: 0.003       # within this of the best -> prefer the lower lr
      sweep:                     # one knob at a time is recommended
        lr: [1.0e-4, 3.0e-4, 1.0e-3]
      epochs: 3                  # any other key = override for pilot runs only
      warmup: 0.3

Each sweep value is a normal `src.train.train` run (same run folder layout),
on a pilot data root whose train.txt / val.txt hold the class subset. Class ids
are kept as-is, so the head stays 10,000-way and prototype init still lines up.

Writes to <output_dir>/pilot_<method>_<img_size>_<timestamp>/:
    pilot_sweep.csv      one row per run (best / final holdout acc, early dip, time)
    pilot_summary.json   chosen values and why
    best_config.yaml     the FULL-run config with the chosen values filled in
                         (pilot overrides removed); run it with src.train
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.train import train
from src.utils import apply_overrides, load_config, save_config

PILOT_META_KEYS = {"num_classes_subset", "holdout_per_class", "tie_tolerance", "sweep", "subset_seed"}


def parse_args():
    p = argparse.ArgumentParser(description="Short pilot runs to choose parameters.")
    p.add_argument("--config", required=True)
    p.add_argument("--data_root", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--pilot_data_dir", default="/content/pilot_data",
                   help="Local folder for the subset label files (images are symlinked, not copied)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override any config key for pilot AND best_config, e.g. --set img_size=336")
    return p.parse_args()


def parse_set(pairs):
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if not key or not _:
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def read_label_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                rows.append((parts[0], int(parts[1])))
    return rows


def build_pilot_root(data_root, pilot_dir, n_classes, holdout_per_class, seed, class_block=1):
    """Subset train.txt by class, carve a labelled holdout, symlink everything else.

    class_block > 1 draws the classes as blocks of that many consecutive class IDs (e.g.
    100 blocks of 10), so related species (neighbouring IDs) are trained together."""
    data_root = Path(data_root)
    block = int(class_block)
    root = Path(pilot_dir) / (f"c{n_classes}_h{holdout_per_class}_s{seed}" + (f"_b{block}" if block > 1 else ""))
    marker = root / ".done"
    if marker.exists():
        print(f"[pilot] reusing pilot data root {root}")
        return root, json.loads(marker.read_text())

    root.mkdir(parents=True, exist_ok=True)
    rows = read_label_file(data_root / "train.txt")
    by_class = defaultdict(list)
    for path, label in rows:
        by_class[label].append(path)

    rng = random.Random(seed)
    classes = sorted(by_class)
    if n_classes < len(classes) and block > 1:
        starts = list(range(0, len(classes) - block + 1, block))
        picked = sorted(rng.sample(starts, n_classes // block))
        classes = [classes[s + j] for s in picked for j in range(block)]
    elif n_classes < len(classes):
        classes = sorted(rng.sample(classes, n_classes))

    fit, hold = [], []
    for c in classes:
        paths = sorted(by_class[c])
        rng.shuffle(paths)
        k = min(holdout_per_class, max(len(paths) - 1, 0))
        hold += [(p, c) for p in paths[:k]]
        fit += [(p, c) for p in paths[k:]]

    for name, items in (("train.txt", fit), ("val.txt", hold)):
        with (root / name).open("w", encoding="utf-8") as f:
            for p, c in items:
                f.write(f"{p} {c}\n")

    # Images and any other files the loader expects keep their relative paths.
    for entry in data_root.iterdir():
        if entry.name in ("train.txt", "val.txt"):
            continue
        link = root / entry.name
        if not link.exists():
            link.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())

    info = {"classes": len(classes), "fit_images": len(fit), "holdout_images": len(hold)}
    marker.write_text(json.dumps(info))
    print(f"[pilot] built {root}: {info}")
    return root, info


def _number_or_text(v):
    try:
        return float(v)
    except ValueError:
        return v  # e.g. img_size "224->336" on the epoch progressive resizing switches


def read_epoch_log(run_dir):
    rows = []
    with (Path(run_dir) / "epoch_log.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["val_acc"] != "":
                rows.append({k: _number_or_text(v) for k, v in r.items() if v != ""})
    return rows


def read_run_stats(run_dir):
    stats = {}
    path = Path(run_dir) / "run_stats.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            k, _, v = line.partition("=")
            stats[k] = v
    return stats


def summarise_run(run_dir, values):
    log = read_epoch_log(run_dir)
    if not log:
        raise RuntimeError(f"No holdout accuracy in {run_dir}/epoch_log.csv (is val.txt labelled?)")
    accs = [r["val_acc"] for r in log]
    best_i = max(range(len(accs)), key=accs.__getitem__)
    # An accuracy drop of more than 0.5 point between epochs usually means the lr is too high.
    dip = any(b < a - 0.005 for a, b in zip(accs, accs[1:]))
    stats = read_run_stats(run_dir)
    return {
        **values,
        "best_val_acc": round(accs[best_i], 4),
        "best_epoch": int(log[best_i]["epoch"]),
        "final_val_acc": round(accs[-1], 4),
        # prediction within 5 class IDs of the label (older logs lack the column)
        "final_val_acc_near5": round(float(log[-1].get("val_acc_near5", float("nan"))), 4),
        "final_val_loss": round(log[-1]["val_loss"], 4),
        "early_dip": dip,
        "trainable_pct": round(float(stats.get("trainable_pct", "nan")), 3),
        "peak_vram_gb": round(float(stats.get("peak_vram_gb", "nan")), 2),
        "wall_min": round(float(stats.get("wall_s", "nan")) / 60, 1),
        "run_dir": str(run_dir),
    }


def choose(results, keys, tie_tol):
    """Best final holdout acc; runs with an early dip lose ties; within tie_tol prefer lower lr."""
    top = max(r["final_val_acc"] for r in results)
    near = [r for r in results if r["final_val_acc"] >= top - tie_tol]
    clean = [r for r in near if not r["early_dip"]] or near
    if "lr" in keys:
        pick = min(clean, key=lambda r: r["lr"])
        reason = f"within {tie_tol} of the best final holdout acc; lowest lr among those"
    else:
        pick = max(clean, key=lambda r: r["final_val_acc"])
        reason = "best final holdout acc"
    if len(clean) < len(near):
        reason += "; skipped runs with an early accuracy dip"
    return pick, reason


def main():
    args = parse_args()
    base = load_config(args.config)
    apply_overrides(base, data_root=args.data_root, output_dir=args.output_dir)
    for k, v in parse_set(args.set).items():
        setattr(base, k, v)

    pilot = dict(base.get("pilot") or {})
    sweep = pilot.get("sweep") or {}
    if not sweep:
        raise ValueError("Config has no pilot.sweep block; nothing to test.")
    if len(sweep) > 1:
        print(f"[pilot] note: sweeping {len(sweep)} knobs jointly; one at a time is cheaper and easier to read")

    n_classes = int(pilot.get("num_classes_subset", 1000))
    holdout = int(pilot.get("holdout_per_class", 5))
    tie_tol = float(pilot.get("tie_tolerance", 0.003))
    subset_seed = int(pilot.get("subset_seed", 0))
    run_overrides = {k: v for k, v in pilot.items() if k not in PILOT_META_KEYS}

    pilot_root, info = build_pilot_root(base.data_root, args.pilot_data_dir, n_classes, holdout, subset_seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(base.output_dir) / f"pilot_{base.method}_{base.get('img_size', 'na')}_{stamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pilot] sweep folder {sweep_dir}")

    keys = list(sweep)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(sweep[k] for k in keys))]
    results = []
    for i, values in enumerate(combos, start=1):
        print(f"\n[pilot] run {i}/{len(combos)}: {values}")
        cfg = load_config(args.config)
        cfg.update(base)
        cfg.update(run_overrides)
        cfg.update(values)
        cfg.data_root = str(pilot_root)
        cfg.output_dir = str(sweep_dir)
        run_dir = train(cfg)
        results.append(summarise_run(run_dir, values))
        del cfg
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (sweep_dir / "pilot_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    pick, reason = choose(results, keys, tie_tol)
    chosen = {k: pick[k] for k in keys}

    full = load_config(args.config)
    full.update(base)
    full.update(chosen)
    full.pilot_result = {"chosen": chosen, "sweep_dir": str(sweep_dir)}
    save_config(full, sweep_dir / "best_config.yaml")

    summary = {"chosen": chosen, "reason": reason, "pilot_data": info,
               "pilot_overrides": run_overrides, "results": results}
    (sweep_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n[pilot] results")
    for r in results:
        flag = "  (early dip)" if r["early_dip"] else ""
        print(f"  {[r[k] for k in keys]}  best={r['best_val_acc']:.4f}@ep{r['best_epoch']}"
              f"  final={r['final_val_acc']:.4f}  {r['wall_min']} min{flag}")
    print(f"[pilot] chosen {chosen} ({reason})")
    print(f"[pilot] full run: python -m src.train --config {sweep_dir / 'best_config.yaml'}")
    return sweep_dir


if __name__ == "__main__":
    main()
