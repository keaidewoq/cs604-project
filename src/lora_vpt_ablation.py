"""Reproducible LoRA + VPT screening; run with python -m src.lora_vpt_ablation.

Subsets remap class IDs to a compact classifier; final runs use original IDs.
Completion markers certify full runs, not resumable optimizer checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch

from src.model import build_model
from src.train import train
from src.utils import Config, load_config, save_config

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "A": ("none", 0), "B": ("shallow", 8), "C": ("shallow", 16),
    "D": ("shallow", 32), "E": ("deep", 8),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def code_provenance():
    files = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "scripts/build_prototypes.py"]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in files}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {"commit": commit, "source_hash": digest(hashes), "files": hashes}


def prepare_split(data_root, destination, classes=1000, holdout=5, seed=42):
    """Persist a deterministic split; absolute image paths need no Drive symlinks."""
    data_root, destination = Path(data_root).resolve(), Path(destination)
    source = data_root / "train.txt"
    wanted = {"source_sha256": file_hash(source), "data_root": str(data_root),
              "classes": classes, "holdout_per_class": holdout, "split_seed": seed}
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "split.json"
    if marker.exists():
        info = json.loads(marker.read_text())
        if info["request"] != wanted:
            raise ValueError("Split settings changed; choose a new experiment directory")
        for name, checksum in info["manifest_hashes"].items():
            if file_hash(destination / name) != checksum:
                raise ValueError(f"Saved split manifest changed: {name}")
        return destination, info
    if classes <= 0 or holdout <= 0:
        raise ValueError("classes and holdout must be positive")
    by_class = defaultdict(list)
    seen = set()
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        path, label = line.rsplit(maxsplit=1)
        absolute = str((data_root / path).resolve())
        if absolute in seen:
            raise ValueError(f"Duplicate training image: {path}")
        seen.add(absolute)
        by_class[int(label)].append(absolute)
    if classes > len(by_class):
        raise ValueError(f"Requested {classes} classes; dataset has {len(by_class)}")
    rng = random.Random(seed)
    selected = sorted(rng.sample(sorted(by_class), classes))
    fitting, validation = [], []
    for local_label, original_label in enumerate(selected):
        paths = sorted(by_class[original_label])
        if len(paths) != 45:
            raise ValueError(f"Class {original_label}: expected 45 images, got {len(paths)}")
        if holdout >= len(paths):
            raise ValueError("Holdout leaves no fitting images")
        rng.shuffle(paths)
        validation.extend((p, local_label) for p in paths[:holdout])
        fitting.extend((p, local_label) for p in paths[holdout:])
    for name, rows in (("train.txt", fitting), ("val.txt", validation)):
        (destination / name).write_text("".join(f"{p} {y}\n" for p, y in rows))
    write_json(destination / "class_map.json", {str(i): c for i, c in enumerate(selected)})
    info = {"request": wanted, "fit_images": len(fitting), "holdout_images": len(validation),
            "manifest_hashes": {n: file_hash(destination / n)
                                for n in ("train.txt", "val.txt", "class_map.json")}}
    write_json(marker, info)
    return destination, info


def ensure_prototypes(cfg, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "prototypes.pt"
    meta = directory / "provenance.json"
    wanted = {"model_name": cfg.model_name, "img_size": cfg.img_size,
              "num_classes": cfg.num_classes, "seed": cfg.seed,
              "fit_manifest_sha256": file_hash(Path(cfg.data_root) / "train.txt"),
              "builder_sha256": file_hash(ROOT / "scripts/build_prototypes.py"),
              "source_hash": code_provenance()["source_hash"]}
    if output.exists() and meta.exists():
        saved = json.loads(meta.read_text())
        if saved["request"] != wanted or saved["sha256"] != file_hash(output):
            raise ValueError("Prototype provenance changed; use a new experiment directory")
        return output
    extraction = Config(cfg.to_dict())
    extraction.update({"method": "linear_probe", "head_init": "random",
                       "prompt_mode": "none", "prompt_length": 0,
                       "prototype_per_class": None})
    config_path = directory / "extraction_config.yaml"
    save_config(extraction, config_path)
    subprocess.run([sys.executable, "-u", "scripts/build_prototypes.py", "--config",
                    str(config_path.resolve()), "--output", str(output.resolve()),
                    "--batch_size", str(cfg.batch_size)], cwd=ROOT, check=True)
    write_json(meta, {"request": wanted, "sha256": file_hash(output)})
    return output


def arm_config(base, arm, seed, prompt_lr):
    cfg = Config(base.to_dict())
    mode, length = ARMS[arm]
    cfg.update({"prompt_mode": mode, "prompt_length": length,
                "prompt_lr": prompt_lr, "seed": seed})
    return cfg


def parameter_stats(cfg):
    check = Config(cfg.to_dict())
    check.pretrained = False
    model = build_model(check)
    stats = dict(model.adaptation_stats)
    stats["prompt_parameters"] = sum(p.numel() for p in model.vpt.parameters()) if hasattr(model, "vpt") else 0
    del model
    gc.collect()
    return stats


def run_one(cfg, directory, context, arm):
    """Skip only completed experiments with identical inputs and source code."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cfg = Config(cfg.to_dict())
    cfg.output_dir = str(directory.resolve())
    request = {"config": cfg.to_dict(), "context": context,
               "prototype_sha256": file_hash(cfg.prototype_file)}
    signature = digest(request)
    marker = directory / "complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["signature"] != signature:
            raise ValueError(f"Completed run settings changed at {directory}; use a new experiment directory")
        run_dir = Path(saved["result"]["run_dir"])
        expected = saved["artifact_hashes"]
        if not all((run_dir / name).is_file() and file_hash(run_dir / name) == checksum
                   for name, checksum in expected.items()):
            raise ValueError(f"Completed run has missing or changed artifacts: {run_dir}")
        print(f"[ablation] reusing {directory}", flush=True)
        return saved["result"]
    write_json(directory / "request.json", request)
    params = parameter_stats(cfg)
    print(f"[ablation] arm={arm} seed={cfg.seed} prompt_lr={cfg.prompt_lr}: {params}", flush=True)
    run_dir = train(cfg)
    rows = list(csv.DictReader((run_dir / "epoch_log.csv").open()))
    raw = dict(line.split("=", 1) for line in (run_dir / "run_stats.txt").read_text().splitlines())
    if cfg.epochs is not None:
        if len(rows) != cfg.epochs or any(r["epoch_complete"] != "1" for r in rows):
            raise RuntimeError("Screening run did not complete every planned epoch")
    elif raw["time_limit_reached"] != "True":
        raise RuntimeError("Final run stopped before its training budget")
    labelled = [r for r in rows if r["val_acc"] != ""]
    best = max(labelled, key=lambda r: float(r["val_acc"])) if labelled else None
    seconds = float(raw["training_seconds"])
    result = {"arm": arm, "seed": cfg.seed, "prompt_mode": cfg.prompt_mode,
              "prompt_length": cfg.prompt_length, "prompt_lr": cfg.prompt_lr,
              "best_val_acc": float(best["val_acc"]) if best else None,
              "best_epoch": int(best["epoch"]) if best else None,
              "final_val_acc": float(rows[-1]["val_acc"]) if labelled else None,
              "final_val_loss": float(rows[-1]["val_loss"]) if labelled else None,
              **params, "peak_vram_GiB": float(raw["peak_vram_gb"]),
              "training_minutes": seconds / 60, "wall_gpu_hours": float(raw["gpu_hours"]),
              "training_images_per_second": sum(int(r["images_processed"]) for r in rows) / seconds,
              "run_dir": str(run_dir.resolve())}
    artifacts = ["last.pt", "epoch_log.csv", "step_log.csv", "run_stats.txt", "config.yaml"]
    if best:
        artifacts.append("best.pt")
    write_json(run_dir / "parameter_stats.json", params)
    write_json(marker, {"signature": signature, "result": result,
                        "artifact_hashes": {n: file_hash(run_dir / n) for n in artifacts}})
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def choose_prompt_lr(results, tolerance=0.003):
    top = max(r["best_val_acc"] for r in results)
    return min(r["prompt_lr"] for r in results if top - r["best_val_acc"] <= tolerance + 1e-12)


def write_comparison(results, destination):
    baselines = {r["seed"]: r["best_val_acc"] for r in results if r["arm"] == "A"}
    rows = []
    for result in results:
        base_acc = baselines.get(result["seed"])
        rows.append({**result, "delta_vs_A_pp":
                     100 * (result["best_val_acc"] - base_acc) if base_acc is not None else None})
    destination = Path(destination)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(destination.with_suffix(".json"), rows)
    return rows


def screen(args):
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    split, info = prepare_split(args.data_root, out / "split", args.classes, args.holdout, args.split_seed)
    base = load_config(args.config)
    base.update({"data_root": str(split), "num_classes": args.classes, "epochs": args.epochs,
                 "warmup": args.epochs * 0.05, "max_train_minutes": None,
                 "schedule_fit_time": False, "seed": args.split_seed,
                 "num_workers": args.workers, "batch_size": args.batch_size,
                 "grad_accum_steps": args.accum})
    base.prototype_file = str(ensure_prototypes(base, out / "prototypes"))
    context = {"split": info, "code": code_provenance(), "torch": str(torch.__version__),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
    write_json(out / "provenance.json", context)
    prompt_lr = args.prompt_lr
    if args.lr_pilot:
        pilots = []
        for rate in (3e-4, 1e-3):
            cfg = arm_config(base, "B", args.seeds[0], rate)
            cfg.epochs, cfg.warmup = args.pilot_epochs, args.pilot_epochs * 0.05
            pilots.append(run_one(cfg, out / "pilot" / f"B_lr{rate}", context, "B"))
        prompt_lr = choose_prompt_lr(pilots)
        write_json(out / "prompt_lr_selection.json", {"selected": prompt_lr,
                   "tie_tolerance": 0.003, "results": pilots})
    write_json(out / f"{args.comparison_stem}_settings.json", {"prompt_lr": prompt_lr, "arms": args.arms,
               "seeds": args.seeds, "epochs": args.epochs})
    # Validate all enabled architectures before starting any screen run.
    for arm in args.arms:
        print(arm, parameter_stats(arm_config(base, arm, args.seeds[0], prompt_lr)), flush=True)
    results = []
    for seed in args.seeds:
        for arm in args.arms:
            cfg = arm_config(base, arm, seed, prompt_lr)
            results.append(run_one(cfg, out / "screen" / f"{arm}_seed{seed}", context, arm))
            write_comparison(results, out / f"{args.comparison_stem}.csv")
    return results


def final_run(args):
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = arm_config(load_config(args.config), args.arm, args.seed, args.prompt_lr)
    # All-data final training is a fresh pretrained start. Never load a pilot.
    cfg.update({"data_root": str(Path(args.data_root).resolve()), "num_classes": 10000,
                "pretrained": True, "epochs": None, "max_train_minutes": args.minutes,
                "time_warmup_fraction": 0.05, "warmup": 0.0, "schedule_fit_time": False,
                "num_workers": args.workers, "batch_size": args.batch_size,
                "grad_accum_steps": args.accum})
    labels = defaultdict(int)
    for line in (Path(cfg.data_root) / "train.txt").read_text().splitlines():
        if line.strip():
            labels[int(line.rsplit(maxsplit=1)[1])] += 1
    if set(labels) != set(range(10000)) or any(n != 45 for n in labels.values()):
        raise ValueError("Final training requires original IDs 0..9999 and 45 images/class")
    # Course validation must be unlabelled; a labelled internal split is invalid here.
    val = Path(cfg.data_root) / "val.txt"
    if val.exists() and any(len(line.split()) > 1 for line in val.read_text().splitlines() if line.strip()):
        raise ValueError("Final data root contains labelled validation; select the official all-data root")
    cfg.prototype_file = str(ensure_prototypes(cfg, out / "prototypes"))
    context = {"code": code_provenance(), "fit_manifest_sha256":
               file_hash(Path(cfg.data_root) / "train.txt"), "torch": str(torch.__version__),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
    result = run_one(cfg, out / f"{args.arm}_seed{args.seed}", context, args.arm)
    write_json(out / "final_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("screen", "final"):
        p = commands.add_parser(name)
        p.add_argument("--config", default="configs/lora_vpt_ablation.yaml")
        p.add_argument("--data-root", required=True)
        p.add_argument("--output-dir", required=True)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--batch-size", type=int, default=128)
        p.add_argument("--accum", type=int, default=2)
        p.add_argument("--prompt-lr", type=float, default=1e-3)
        if name == "screen":
            p.add_argument("--classes", type=int, default=1000)
            p.add_argument("--holdout", type=int, default=5)
            p.add_argument("--split-seed", type=int, default=42)
            p.add_argument("--epochs", type=int, default=5)
            p.add_argument("--seeds", nargs="+", type=int, default=[42])
            p.add_argument("--arms", nargs="+", choices=ARMS, default=["A", "B", "E", "C", "D"])
            p.add_argument("--lr-pilot", action="store_true")
            p.add_argument("--pilot-epochs", type=int, default=3)
            p.add_argument("--comparison-stem", default="comparison")
        else:
            p.add_argument("--arm", choices=ARMS, required=True)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--minutes", type=float, default=144)
    args = parser.parse_args()
    if args.command == "screen" and (args.epochs < 1 or args.pilot_epochs < 1):
        parser.error("Epoch counts must be positive")
    if args.batch_size < 1 or args.accum < 1 or args.prompt_lr <= 0:
        parser.error("Batch size, accumulation and prompt learning rate must be positive")
    screen(args) if args.command == "screen" else final_run(args)


if __name__ == "__main__":
    main()
