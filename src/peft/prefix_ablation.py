"""Colab helpers for the controlled prefix-length / placement experiment.

Completed arms are reused only when their configuration and data hashes match.
Interrupted arms restart from the pretrained model, not an incomplete checkpoint.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import timm

from src.heads import load_prototypes
from src.model import build_model
from src.pilot_sweep import choose, read_epoch_log, read_run_stats
from src.train import train, evaluate
from src.data import ImageListDataset, build_transforms, build_dataloaders
from src.losses import build_loss
from src.utils import get_device, set_seed
from torch.utils.data import DataLoader
from torch.nn import functional as F
from src.utils import Config, save_config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def arm_grid(cfg):
    settings = cfg.ablation
    arms = [{"arm": "classifier_only", "method": "linear_probe",
             "prefix_length": 0, "prefix_placement": "none"}]
    for placement in settings["placements"]:
        for length in settings["lengths"]:
            arms.append({"arm": f"{placement}_p{length}", "method": "prefix",
                         "prefix_length": length, "prefix_placement": placement})
    return arms


def prepare(cfg, experiment_dir, split_dir=None):
    """Persist a fixed split, then select pilot classes from that same split."""
    experiment = Path(experiment_dir)
    experiment.mkdir(parents=True, exist_ok=True)
    source = Path(cfg.data_root).resolve()
    settings = cfg.ablation
    identity = {
        "source_manifest_sha256": digest(source / "train.txt"),
        "split_seed": int(settings["split_seed"]),
        "holdout_per_class": int(settings["holdout_per_class"]),
        "num_classes": int(cfg.num_classes),
        "pilot_classes": int(settings["pilot_classes"]),
        "model_name": cfg.model_name, "img_size": int(cfg.img_size),
    }
    marker = experiment / "experiment.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("Experiment identity changed; use a new EXPERIMENT_NAME")
    groups = defaultdict(list)
    seen = set()
    for line in (source / "train.txt").read_text().splitlines():
        if not line.strip():
            continue
        path, label = line.split()
        label = int(label)
        if path in seen:
            raise ValueError(f"Duplicate image in source manifest: {path}")
        seen.add(path)
        groups[label].append(path)
    if set(groups) != set(range(int(cfg.num_classes))):
        raise ValueError("Source manifest must contain every configured class")
    rng = random.Random(identity["split_seed"])
    fitting, holdout = [], []
    for label in sorted(groups):
        paths = sorted(groups[label])
        if len(paths) != 45:
            raise ValueError(f"Class {label} has {len(paths)} images; expected 45")
        rng.shuffle(paths)
        k = identity["holdout_per_class"]
        if not 0 < k < len(paths):
            raise ValueError("holdout_per_class must leave fitting and validation images")
        fitting.extend((p, label) for p in paths[k:])
        holdout.extend((p, label) for p in paths[:k])
    pilot_count = identity["pilot_classes"]
    if not 0 < pilot_count <= len(groups):
        raise ValueError("pilot_classes must be within the dataset class count")
    pilot_classes = set(random.Random(identity["split_seed"]).sample(sorted(groups), pilot_count))
    roots = {}
    for name, classes in (("full", set(groups)), ("pilot", pilot_classes)):
        saved_root = experiment / "splits" / name
        saved_root.mkdir(parents=True, exist_ok=True)
        root = Path(split_dir) / name if split_dir else saved_root
        root.mkdir(parents=True, exist_ok=True)
        for split, rows in (("train", fitting), ("val", holdout)):
            content = "".join(f"{p} {c}\n" for p, c in rows if c in classes)
            dest = saved_root / f"{split}.txt"
            if dest.exists() and dest.read_text() != content:
                raise ValueError(f"Saved split was modified: {dest}")
            dest.write_text(content, encoding="utf-8")
            if root != saved_root:
                (root / f"{split}.txt").write_text(content, encoding="utf-8")
        # Link only directories/files referenced by labelled training images.
        image_roots = {Path(p).parts[0] for p, _ in fitting + holdout if not Path(p).is_absolute()}
        for entry in source.iterdir():
            if entry.name not in image_roots or entry.name in ("train.txt", "val.txt", "test.txt"):
                continue
            link = root / entry.name
            if link.is_symlink() and link.resolve() != entry.resolve():
                link.unlink()
            if not link.exists() and not link.is_symlink():
                link.symlink_to(entry, target_is_directory=entry.is_dir())
        roots[name] = root
    write_json(marker, identity)
    save_config(cfg, experiment / "base_config.yaml")
    write_json(experiment / "arms.json", arm_grid(cfg))
    return roots


def stage_config(base, experiment, root, stage, arm, seed, lr):
    cfg = Config(base.to_dict())
    cfg.update(arm)
    cfg.seed = seed
    cfg.lr = lr
    cfg.data_root = str(root)
    cfg.prototype_file = str(Path(experiment) / f"prototypes_{root.name}_336.pt")
    cfg.output_dir = str(Path(experiment) / "runs" / stage / arm["arm"] / f"seed{seed}")
    cfg.max_train_minutes = None
    cfg.evaluate_before_training = False  # Wrapper records the initial evaluation.
    if stage == "pilot":
        cfg.epochs = int(base.ablation["pilot_epochs"])
    cfg.warmup = 0.05 * int(cfg.epochs)
    return cfg



@torch.no_grad()
def build_prototypes(cfg, output, batch_size=128, allow_missing_classes=False):
    """Build fit-only prototypes, retaining empty rows for absent pilot classes."""
    device = get_device()
    _, transform = build_transforms(cfg)
    dataset = ImageListDataset(cfg.data_root, "train", transform=transform)
    if not dataset.has_labels:
        raise ValueError("Prototype fitting manifest must be labelled")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=int(cfg.num_workers), pin_memory=device.type == "cuda")
    feature_cfg = Config(cfg.to_dict())
    feature_cfg.method = "linear_probe"
    feature_cfg.head_init = "random"
    model = build_model(feature_cfg).to(device).eval()
    sums = torch.zeros(int(cfg.num_classes), model.num_features, device=device)
    counts = torch.zeros(int(cfg.num_classes), device=device)
    from tqdm import tqdm
    for images, labels, _ in tqdm(loader, desc="Fit-only prototypes"):
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda" and bool(cfg.bf16)):
            features = model.forward_head(model.forward_features(images), pre_logits=True)
        sums.index_add_(0, labels, F.normalize(features.float(), dim=1))
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    if (counts == 0).any() and not allow_missing_classes:
        raise ValueError("Full fitting set is missing classes")
    prototypes = F.normalize(sums / counts.clamp_min(1)[:, None], dim=1).cpu()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prototypes": prototypes, "samples_per_class": counts.cpu(),
                "model_name": cfg.model_name, "img_size": int(cfg.img_size),
                "fit_manifest_sha256": digest(Path(cfg.data_root) / "train.txt"),
                "per_class_cap": None}, output)
    print(f"[prototypes] saved {output}")


def ensure_prototypes(cfg, allow_missing_classes=False):
    path = Path(cfg.prototype_file)
    if not path.exists():
        build_prototypes(cfg, path, batch_size=int(cfg.batch_size),
                         allow_missing_classes=allow_missing_classes)
    # Validate metadata without allocating another full model.
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or (
        blob.get("model_name") != cfg.model_name
        or blob.get("img_size") != int(cfg.img_size)
        or blob.get("fit_manifest_sha256") != digest(Path(cfg.data_root) / "train.txt")
        or blob.get("per_class_cap") is not None
        or tuple(blob["prototypes"].shape) != (int(cfg.num_classes), 768)
    ):
        raise ValueError(f"Stale or incompatible prototypes: {path}")
    del blob
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_arm(cfg):
    """Persist completion and exact recipe to safely resume a multi-session grid."""
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    signature = {"config": cfg.to_dict(),
                 "fit_sha256": digest(Path(cfg.data_root) / "train.txt"),
                 "holdout_sha256": digest(Path(cfg.data_root) / "val.txt"),
                 "prototype_sha256": digest(cfg.prototype_file),
                 "torch_version": str(torch.__version__), "timm_version": timm.__version__,
                 "code_sha256": {name: digest(Path(__file__).resolve().parents[2] / name)
                                 for name in ("src/peft/prefix.py", "src/train.py", "src/model.py",
                                              "src/data.py", "src/heads.py", "src/peft/prefix_ablation.py",
                                              "scripts/build_prototypes.py")}}
    recipe = output / "recipe.json"
    if recipe.exists() and json.loads(recipe.read_text()) != signature:
        raise ValueError(f"Recipe changed for {output}; use a new experiment directory")
    write_json(recipe, signature)
    done = output / "completed.json"
    if done.exists():
        result = json.loads(done.read_text())
        if not (Path(result["run_dir"]) / "last.pt").exists():
            raise FileNotFoundError("Completed arm checkpoint is missing")
        print(f"[ablation] reusing {output}")
        return result
    # Match the trainer initialization exactly for pretraining evaluation.
    set_seed(int(cfg.seed))
    model = build_model(cfg)
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    original = sum(p.numel() for n, p in model.named_parameters()
                   if id(p) not in head_ids and not n.endswith(("prefix_key", "prefix_value")))
    adaptation = sum(p.numel() for p in model.parameters()
                     if p.requires_grad and id(p) not in head_ids)
    unexpected = [n for n, p in model.named_parameters()
                  if p.requires_grad and id(p) not in head_ids
                  and not n.endswith(("prefix_key", "prefix_value"))]
    if unexpected or 100 * adaptation / original >= float(cfg.max_adaptation_pct):
        raise ValueError(f"Frozen backbone / adaptation cap violation: {unexpected}")
    load_prototypes(cfg, model)
    device = get_device()
    model = model.to(device)
    loaders = build_dataloaders(cfg)
    before = evaluate(model, loaders["val"], build_loss(cfg), device, bool(cfg.bf16))
    del model, loaders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_dir = train(cfg)
    write_json(run_dir / "before_training.json", before)
    write_json(run_dir / "adaptation_stats.json", {"adaptation_parameters": adaptation,
               "original_backbone_parameters": original, "adaptation_pct": 100 * adaptation / original})
    log = read_epoch_log(run_dir)
    if len(log) != int(cfg.epochs) or any(not r["epoch_complete"] for r in log):
        raise RuntimeError("Incomplete arm cannot be included in the ablation")
    best = max(log, key=lambda r: r["val_acc"])
    stats = read_run_stats(run_dir)
    result = {
        "arm": cfg.arm, "seed": int(cfg.seed), "lr": float(cfg.lr),
        "prefix_length": int(cfg.prefix_length), "prefix_placement": cfg.prefix_placement,
        "before_val_acc": json.loads((run_dir / "before_training.json").read_text())["val_acc"],
        "best_val_acc": best["val_acc"], "best_epoch": int(best["epoch"]),
        "final_val_acc": log[-1]["val_acc"],
        "final_val_loss": log[-1]["val_loss"],
        "early_dip": any(b["val_acc"] < a["val_acc"] - 0.005 for a, b in zip(log, log[1:])),
        "adaptation_parameters": adaptation, "original_backbone_parameters": original,
        "adaptation_pct": 100 * adaptation / original,
        "peak_vram_gb": float(stats["peak_vram_gb"]),
        "training_minutes": float(stats["training_minutes"]),
        "wall_gpu_hours": float(stats["gpu_hours"]),
        "training_images_per_second": sum(r["images_processed"] for r in log) / float(stats["training_seconds"]),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch_version": str(torch.__version__), "run_dir": str(run_dir),
        "timm_version": timm.__version__,
    }
    write_json(done, result)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_results(experiment, stage, results):
    path = Path(experiment) / f"{stage}_results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return path


def run_stage(base, experiment, roots, stage, arms=None, seeds=None, prefix_lr=None):
    """Run pilot, screen, or explicitly selected confirmation arms."""
    if stage not in ("pilot", "screen", "confirm"):
        raise ValueError("stage must be pilot, screen, or confirm")
    experiment = Path(experiment)
    if stage == "pilot":
        selected = [{"arm": f"all_p20_lr{lr:g}", "method": "prefix",
                     "prefix_length": 20, "prefix_placement": "all", "lr": lr}
                    for lr in base.ablation["pilot_lrs"]]
        root = roots["pilot"]
        seeds = [int(base.seed)]
    else:
        root = roots["full"]
        selected = arm_grid(base)
        if arms is not None:
            unknown = set(arms) - {a["arm"] for a in selected}
            if unknown:
                raise ValueError(f"Unknown arms: {unknown}")
            selected = [a for a in selected if a["arm"] in arms]
        if not selected or (stage == "confirm" and arms is None):
            raise ValueError("Confirmation requires explicitly selected arms")
        seeds = seeds or [int(base.seed)]
        if prefix_lr is None:
            chosen_path = experiment / "pilot_selection.json"
            if not chosen_path.exists():
                raise ValueError("Run the LR pilot first or explicitly set PREFIX_LR")
            prefix_lr = json.loads(chosen_path.read_text())["lr"]
    results = []
    for arm in selected:
        for seed in seeds:
            lr = arm.get("lr", prefix_lr)
            run_stage_name = stage
            screen_done = experiment / "runs" / "screen" / arm["arm"] / f"seed{seed}" / "completed.json"
            if stage == "confirm" and seed == int(base.seed) and screen_done.exists():
                run_stage_name = "screen"
            cfg = stage_config(base, experiment, root, run_stage_name, arm, seed, lr)
            ensure_prototypes(cfg, allow_missing_classes=stage == "pilot")
            results.append(run_arm(cfg))
            write_results(experiment, stage, results)
    if stage == "pilot":
        pick, reason = choose(results, ["lr"], float(base.ablation["tie_tolerance"]))
        write_json(experiment / "pilot_selection.json", {"lr": pick["lr"], "reason": reason})
    return results


def export_final_config(base, experiment, selected_arm, prefix_lr=None):
    """Export an all-data 144-minute recipe; do not start final training."""
    arms = {a["arm"]: a for a in arm_grid(base)}
    if selected_arm not in arms or selected_arm == "classifier_only":
        raise ValueError("Select a prefix arm from the declared grid")
    if prefix_lr is None:
        prefix_lr = json.loads((Path(experiment) / "pilot_selection.json").read_text())["lr"]
    cfg = Config(base.to_dict())
    cfg.update(arms[selected_arm])
    cfg.lr = prefix_lr
    cfg.epochs = None
    cfg.max_train_minutes = 144
    cfg.time_warmup_fraction = 0.05
    cfg.prototype_file = str(Path(experiment) / "prototypes_all_data_336.pt")
    cfg.output_dir = str(Path(experiment) / "final")
    cfg.evaluate_before_training = False
    path = Path(experiment) / "final_config.yaml"
    save_config(cfg, path)
    return path
