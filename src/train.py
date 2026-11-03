"""Train entry point: python -m src.train --config configs/X.yaml"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.data import build_dataloaders
from src.heads import load_probe_head, load_prototypes
from src.losses import build_loss
from src.model import build_model
from src.utils import (
    PEFT_METHODS,
    Timer,
    VRAMTracker,
    apply_overrides,
    get_device,
    load_config,
    gpu_hours,
    log_run_stats,
    save_config,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune a ViT with a method config.")
    p.add_argument("--config", default=None, help="Path to a method YAML (e.g. configs/lora_r8.yaml)")
    p.add_argument("--resume", default=None, metavar="RUN_DIR",
                   help="Continue a crashed run from RUN_DIR/last.pt (uses RUN_DIR/config.yaml; --config not needed)")
    p.add_argument("--data_root", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    args = p.parse_args()
    if not args.config and not args.resume:
        p.error("--config is required (or --resume RUN_DIR)")
    return args


def _split_decay(params, weight_decay, lr):
    decay, no_decay = [], []
    for param in params:
        if not param.requires_grad:
            continue
        if param.ndim < 2 or getattr(param, "_no_weight_decay", False):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


def _vit_depth(name, num_blocks):
    """0 for embeddings, i + 1 for blocks.i, num_blocks + 1 for the final norm."""
    if name.startswith(("patch_embed.", "cls_token", "pos_embed", "reg_token")):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return num_blocks + 1


def layer_decay_param_groups(model, weight_decay, lr, head_lr, layer_decay):
    """Layer-wise lr decay (`layer_decay` set): block i gets lr * layer_decay ** (blocks - i).

    The final norm keeps `lr`, the embeddings get the smallest rate, and the
    classifier gets `head_lr` (or `lr`). The top backbone group comes first, so
    the logged lr is the top backbone rate.
    """
    num_blocks = len(model.blocks)
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    by_depth = {}
    head = []
    for name, param in model.named_parameters():
        if id(param) in head_ids:
            head.append(param)
        else:
            by_depth.setdefault(_vit_depth(name, num_blocks), []).append(param)
    groups = []
    for depth in sorted(by_depth, reverse=True):
        scale = float(layer_decay) ** (num_blocks + 1 - depth)
        groups.extend(_split_decay(by_depth[depth], weight_decay, lr * scale))
    groups.extend(_split_decay(head, weight_decay, float(head_lr if head_lr is not None else lr)))
    print(f"[optim] layer_decay={layer_decay} top lr={lr} bottom lr={lr * float(layer_decay) ** (num_blocks + 1):.2e} head_lr={head_lr}")
    return groups


def adamw_param_groups(model, weight_decay, lr, head_lr=None, layer_decay=None, unfreeze_lr=None, prompt_lr=None):
    """One lr for everything, or backbone `lr` + classifier `head_lr` (Qian's FT).

    With `unfreeze_lr`, pretrained weights unfrozen by src/peft/selected.py
    (marked `_unfrozen`) get their own lr; the rest of the backbone keeps `lr`.
    VPT prompts (src/peft/vpt.py) get `prompt_lr`.
    """
    prompts = list(model.vpt.parameters()) if hasattr(model, "vpt") else []
    if prompts:
        if layer_decay is not None:
            raise ValueError("VPT with layer_decay is not supported")
        head_ids = {id(p) for p in model.get_classifier().parameters()}
        prompt_ids = {id(p) for p in prompts}
        backbone = [p for p in model.parameters() if id(p) not in head_ids | prompt_ids]
        groups = _split_decay(backbone, weight_decay, lr)
        groups.extend(_split_decay(model.get_classifier().parameters(), weight_decay,
                                   float(head_lr if head_lr is not None else lr)))
        groups.extend(_split_decay(prompts, weight_decay,
                                   float(prompt_lr if prompt_lr is not None else lr)))
        return groups
    if layer_decay is not None:
        return layer_decay_param_groups(model, weight_decay, lr, head_lr, layer_decay)
    if head_lr is None and unfreeze_lr is None:
        return _split_decay(model.parameters(), weight_decay, lr)
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    unfrozen = []
    if unfreeze_lr is not None:
        unfrozen = [p for p in model.parameters() if id(p) not in head_ids and getattr(p, "_unfrozen", False)]
    unfrozen_ids = {id(p) for p in unfrozen}
    backbone = [p for p in model.parameters() if id(p) not in head_ids and id(p) not in unfrozen_ids]
    head = [p for p in model.parameters() if id(p) in head_ids]
    groups = _split_decay(backbone, weight_decay, lr)
    groups.extend(_split_decay(unfrozen, weight_decay, float(unfreeze_lr or lr)))
    groups.extend(_split_decay(head, weight_decay, float(head_lr if head_lr is not None else lr)))
    if not groups:
        raise ValueError("No trainable parameters")
    extra = f"  unfreeze_lr={unfreeze_lr} ({len(unfrozen)} tensors)" if unfrozen else ""
    print(f"[optim] backbone lr={lr}  head_lr={head_lr}{extra}")
    return groups


def build_scheduler(optimizer, warmup_steps, total_steps, clock=None):
    """Warmup then cosine to zero over `total_steps`.

    With `clock` (schedule_fit_time), the position is the larger of the step
    count and the fraction of max_train_minutes used, so the cosine still
    reaches zero when the time limit stops training before `epochs`.
    """

    def lr_lambda(step):
        if clock is not None:
            step = max(step, min(clock["fraction"], 1.0) * total_steps)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def collect_state(model, method):
    if method in PEFT_METHODS:
        return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, cfg, epoch, metrics, resume=None):
    """Weights (trainable only for PEFT). With `resume`, also what --resume needs to continue.

    Written to a temporary file first, so a crash during the save leaves the
    previous checkpoint intact.
    """
    payload = {
        "state_dict": collect_state(model, cfg.method),
        "epoch": epoch,
        "metrics": metrics or {},
        "method": cfg.method,
        "model_name": cfg.model_name,
        "num_classes": cfg.num_classes,
        "trainable_only": cfg.method in PEFT_METHODS,
    }
    if resume is not None:
        payload["resume"] = resume
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _trim_log(path, last_epoch):
    """Keep the header and rows with epoch <= last_epoch (drop a crashed epoch's partial rows)."""
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return False
    epoch_col = rows[0].index("epoch")
    kept = [rows[0]] + [r for r in rows[1:] if r and int(r[epoch_col]) <= last_epoch]
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(kept)
    return True


def amp_context(device, use_bf16):
    enabled = bool(use_bf16) and device.type == "cuda"
    dtype = torch.bfloat16 if enabled else torch.float32
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_bf16):
    """Holdout loss and top-1, plus `val_acc_near5`: prediction within 5 class IDs of the
    label (IDs follow the taxonomy here, so roughly "right genus")."""
    model.eval()
    total_loss = 0.0
    correct = 0
    near = 0
    n = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_context(device, use_bf16):
            logits = model(images)
            loss = loss_fn(logits, labels)
        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        near += ((pred - labels).abs() <= 5).sum().item()
        n += labels.size(0)
    return {"val_loss": total_loss / max(n, 1), "val_acc": correct / max(n, 1),
            "val_acc_near5": near / max(n, 1)}


def train(cfg, resume_dir=None):
    """Train from scratch, or with `resume_dir` continue that run from its last.pt.

    A resumed run restores the weights, optimizer, lr schedule, training clock and
    RNG state saved at the end of the last finished epoch, and continues in the same
    folder with the remaining time budget. Work done in a crashed epoch is redone.
    """
    max_train_minutes = cfg.get("max_train_minutes")
    max_train_seconds = None
    if max_train_minutes is not None:
        max_train_seconds = float(max_train_minutes) * 60
        if not math.isfinite(max_train_seconds) or max_train_seconds <= 0:
            raise ValueError("max_train_minutes must be finite and positive")
    set_seed(int(cfg.seed))
    device = get_device()
    print(f"[train] device={device} method={cfg.method} model={cfg.model_name}")

    ckpt = None
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        ckpt = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        if "resume" not in ckpt:
            raise SystemExit(f"{run_dir / 'last.pt'} was saved without resume state "
                             "(before resume support, or save_resume_state: false); rerun from scratch")
        if ckpt["resume"].get("finished"):
            raise SystemExit(f"{run_dir} already finished training; nothing to resume")
        print(f"[train] resuming run_dir={run_dir} after epoch {ckpt['epoch']}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg.output_dir) / f"{cfg.method}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, run_dir / "config.yaml")
        print(f"[train] run_dir={run_dir}")

    timer = Timer()
    vram = VRAMTracker()

    loaders = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    load_prototypes(cfg, model)
    load_probe_head(cfg, model)
    if ckpt is not None:
        _, unexpected = model.load_state_dict(ckpt["state_dict"], strict=not ckpt["trainable_only"])
        if unexpected:
            raise ValueError(f"checkpoint has keys this config does not build: {unexpected[:5]}")
    loss_fn = build_loss(cfg, model)

    groups = adamw_param_groups(
        model, float(cfg.weight_decay), float(cfg.lr), head_lr=cfg.get("head_lr"),
        layer_decay=cfg.get("layer_decay"), unfreeze_lr=cfg.get("unfreeze_lr"),
        prompt_lr=cfg.get("prompt_lr"),
    )
    # Opt-in (Van): fused CUDA AdamW kernel, same update rule; off keeps the default kernel.
    adamw_kwargs = {"fused": True} if bool(cfg.get("fused_adamw", False)) else {}
    optimizer = AdamW(groups, lr=float(cfg.lr), **adamw_kwargs)
    # Opt-in (Van): torch.compile for the training forward/backward only. `model` stays
    # the plain module, so checkpoints, evaluation and PCA init are unchanged. Compiling
    # happens in the first training steps, so its time counts against the time limit.
    train_model = torch.compile(model) if bool(cfg.get("torch_compile", False)) else model
    if adamw_kwargs or train_model is not model:
        print(f"[train] fused_adamw={bool(adamw_kwargs)} torch_compile={train_model is not model}")

    accum = max(int(cfg.grad_accum_steps), 1)
    steps_per_epoch = math.ceil(len(loaders["train"]) / accum)
    time_only = cfg.get("epochs") is None
    elapsed_training = [0.0]
    clock = None
    total_steps = None
    if time_only:
        if max_train_seconds is None:
            raise ValueError("epochs: null requires max_train_minutes")
        warmup_fraction = float(cfg.get("time_warmup_fraction", 0.0))
        if not 0 <= warmup_fraction < 1:
            raise ValueError("time_warmup_fraction must be in [0, 1)")
        def time_lr(_step):
            progress = min(max(elapsed_training[0] / max_train_seconds, 0.0), 1.0)
            if warmup_fraction and progress < warmup_fraction:
                return progress / warmup_fraction
            decay = (progress - warmup_fraction) / (1 - warmup_fraction)
            return 0.5 * (1 + math.cos(math.pi * decay))
        scheduler = LambdaLR(optimizer, time_lr)
    else:
        total_steps = max(steps_per_epoch * int(cfg.epochs), 1)
        warmup_steps = int(float(cfg.warmup) * steps_per_epoch)
        if bool(cfg.get("schedule_fit_time", False)):
            if max_train_seconds is None:
                raise ValueError("schedule_fit_time needs max_train_minutes")
            clock = {"fraction": 0.0}
        scheduler = build_scheduler(optimizer, warmup_steps, total_steps, clock)
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["resume"]["optimizer"])
        scheduler.load_state_dict(ckpt["resume"]["scheduler"])
        if clock is not None:
            clock["fraction"] = ckpt["resume"]["clock_fraction"]
        elapsed_training[0] = ckpt["resume"]["elapsed_training"]
    clip_grad_norm = cfg.get("clip_grad_norm")
    # Progressive resizing (FixRes-style): train at low_res_size until training is
    # low_res_fraction done, then at img_size. "Done" is the same position the lr
    # schedule uses: the larger of the step fraction and the time-budget fraction.
    low_res_fraction = None
    if loaders.get("train_low") is not None:
        low_res_fraction = float(cfg.get("low_res_fraction", 0.75))
        if not 0 < low_res_fraction < 1:
            raise ValueError("low_res_fraction must be in (0, 1)")
        print(f"[train] progressive resize: {cfg.low_res_size}px for the first {low_res_fraction:.0%}, "
              f"then {cfg.img_size}px")
    if len(loaders["train"]) == 0:
        raise ValueError("Training loader has no batches; reduce batch_size")
    print(f"[train] images={len(loaders['train'].dataset)}, batches_per_epoch={len(loaders['train'])}, "
          f"batch_size={cfg.batch_size}, workers={cfg.num_workers}")
    if device.type == "cuda":
        print(f"[train] GPU={torch.cuda.get_device_name(device)}")
    else:
        print("[train] WARNING: running on CPU")
    print(f"[train] stopping={'training-time budget only' if time_only else 'epoch ceiling or time budget'}")


    log_run_stats("start", model, timer, vram)
    vram.reset()

    has_val = bool(loaders["val"] is not None and loaders["has_val_labels"])
    use_bf16 = bool(cfg.bf16)
    if has_val and ckpt is None and bool(cfg.get("evaluate_before_training", False)):
        before = evaluate(model, loaders["val"], loss_fn, device, use_bf16)
        print(
            f"[before train] val_loss={before['val_loss']:.4f} val_acc={before['val_acc']:.4f} "
            "(prototype / random head, no updates yet)"
        )
        if bool(cfg.get("evaluate_only", False)):
            save_checkpoint(run_dir / "last.pt", model, cfg, 0, before)
            stats = log_run_stats("end", model, timer, vram)
            with (run_dir / "run_stats.txt").open("w", encoding="utf-8") as f:
                for k, v in stats.items():
                    f.write(f"{k}={v}\n")
            print(f"[train] evaluate_only done. {run_dir}")
            return run_dir

    step_path = run_dir / "step_log.csv"
    epoch_path = run_dir / "epoch_log.csv"
    append = False
    if ckpt is not None:
        append = _trim_log(epoch_path, ckpt["epoch"]) & _trim_log(step_path, ckpt["epoch"])
    step_f = step_path.open("a" if append else "w", newline="", encoding="utf-8")
    epoch_f = epoch_path.open("a" if append else "w", newline="", encoding="utf-8")
    step_w = csv.DictWriter(step_f, fieldnames=["step", "epoch", "loss", "lr"])
    epoch_fields = ["epoch", "train_loss", "val_loss", "val_acc", "val_acc_near5", "epoch_time_s",
                    "batches_processed", "batches_total", "images_processed", "epoch_complete", "img_size"]
    if append:  # a resumed run keeps the columns its log started with
        with epoch_path.open(newline="", encoding="utf-8") as f:
            epoch_fields = next(csv.reader(f))
    epoch_w = csv.DictWriter(epoch_f, fieldnames=epoch_fields, extrasaction="ignore")
    if not append:
        step_w.writeheader()
        epoch_w.writeheader()

    best_acc = -1.0
    global_step = 0
    training_seconds = 0.0
    time_limit_reached = False
    start_epoch = 1
    prior_wall_s = 0.0
    prior_peak_vram_gb = 0.0
    save_resume = bool(cfg.get("save_resume_state", False))  # opt-in: optimizer state makes last.pt ~3x larger for full FT
    if ckpt is not None:
        r = ckpt["resume"]
        best_acc = r["best_acc"]
        global_step = r["global_step"]
        training_seconds = r["training_seconds"]
        prior_wall_s = r["wall_s"]
        prior_peak_vram_gb = r["peak_vram_gb"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[train] resumed at step {global_step}, {training_seconds / 60:.2f} training minutes used, "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
        set_rng_state(r["rng"])  # same shuffling order as an uninterrupted run
        del ckpt
    if max_train_seconds is not None:
        print(f"[train] training-only time limit={max_train_minutes} minutes")

    def progress(training_start):
        done = global_step / total_steps if total_steps else 0.0
        if max_train_seconds is not None:
            used = training_seconds + time.monotonic() - training_start
            done = max(done, used / max_train_seconds)
        return done

    try:
        epochs = itertools.count(start_epoch) if time_only else range(start_epoch, int(cfg.epochs) + 1)
        for epoch in epochs:
            model.train()
            epoch_timer = Timer()
            running = 0.0
            seen = 0
            batches_processed = 0
            optimizer.zero_grad(set_to_none=True)

            # Count training batch loading and updates, excluding setup,
            # prototype loading, validation, and checkpoint saving.
            if max_train_seconds is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            training_start = time.monotonic()
            low_res = low_res_fraction is not None and progress(training_start) < low_res_fraction
            epoch_sizes = [int(cfg.low_res_size) if low_res else int(cfg.img_size)]
            n_batches = len(loaders["train"])

            def epoch_batches():
                if not low_res:
                    yield from loaders["train"]
                    return
                # Low-res batches until the phase is over (checked at accumulation
                # boundaries), then the rest of the epoch from the img_size loader.
                k = 0
                for k, batch in enumerate(loaders["train_low"]):
                    if k % accum == 0 and progress(training_start) >= low_res_fraction:
                        break
                    yield batch
                else:
                    return
                print(f"\n[train] switching to {cfg.img_size}px after {global_step} steps")
                epoch_sizes.append(int(cfg.img_size))
                yield from itertools.islice(loaders["train"], n_batches - k)

            label = f"epoch {epoch} (time budget)" if time_only else f"epoch {epoch}/{cfg.epochs}"
            pbar = tqdm(epoch_batches(), total=n_batches, desc=f"{label} @{epoch_sizes[0]}px", leave=True)
            for i, (images, labels, _) in enumerate(pbar, start=1):
                if (max_train_seconds is not None and
                        training_seconds + time.monotonic() - training_start >= max_train_seconds):
                    time_limit_reached = True
                    # Discard any incomplete gradient accumulation group.
                    optimizer.zero_grad(set_to_none=True)
                    break
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with amp_context(device, use_bf16):
                    logits = train_model(images)
                    loss = loss_fn(logits, labels)
                (loss / accum).backward()

                if i % accum == 0 or i == n_batches:
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], float(clip_grad_norm)
                        )
                    optimizer.step()
                    if clock is not None:
                        used = training_seconds + time.monotonic() - training_start
                        clock["fraction"] = used / max_train_seconds
                    if time_only:
                        elapsed_training[0] = training_seconds + time.monotonic() - training_start
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                loss_val = float(loss.item())
                running += loss_val * labels.size(0)
                seen += labels.size(0)
                batches_processed += 1
                lr = scheduler.get_last_lr()[0]
                step_w.writerow(
                    {"step": global_step, "epoch": epoch, "loss": loss_val, "lr": lr}
                )
                pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}")

            pbar.close()
            if max_train_seconds is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            training_seconds += time.monotonic() - training_start
            if max_train_seconds is not None and training_seconds >= max_train_seconds:
                time_limit_reached = True
            step_f.flush()

            row = {
                "epoch": epoch,
                "train_loss": running / max(seen, 1),
                "val_loss": "",
                "val_acc": "",
                "epoch_time_s": round(epoch_timer.elapsed(), 2),
                "batches_processed": batches_processed,
                "batches_total": n_batches,
                "images_processed": seen,
                "epoch_complete": int(batches_processed == n_batches),
                "img_size": "->".join(map(str, epoch_sizes)),
            }
            if has_val:
                metrics = evaluate(model, loaders["val"], loss_fn, device, use_bf16)
                row["val_loss"] = metrics["val_loss"]
                row["val_acc"] = metrics["val_acc"]
                row["val_acc_near5"] = metrics["val_acc_near5"]
                print(
                    f"[epoch {epoch}] train_loss={row['train_loss']:.4f} "
                    f"val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_acc']:.4f} "
                    f"near5={metrics['val_acc_near5']:.4f}"
                )
                if metrics["val_acc"] > best_acc:
                    best_acc = metrics["val_acc"]
                    save_checkpoint(run_dir / "best.pt", model, cfg, epoch, metrics)
                    print(f"[epoch {epoch}] new best val_acc={best_acc:.4f}")
            else:
                print(f"[epoch {epoch}] train_loss={row['train_loss']:.4f} (no labeled val)")

            epoch_w.writerow(row)
            epoch_f.flush()
            resume = None
            if save_resume:
                finished = time_limit_reached or (not time_only and epoch == int(cfg.epochs))
                resume = {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "clock_fraction": clock["fraction"] if clock is not None else None,
                    "elapsed_training": elapsed_training[0],
                    "global_step": global_step,
                    "training_seconds": training_seconds,
                    "best_acc": best_acc,
                    "wall_s": prior_wall_s + timer.elapsed(),
                    "peak_vram_gb": max(prior_peak_vram_gb, vram.peak_gb()),
                    "rng": rng_state(),
                    "finished": finished,
                }
            save_checkpoint(run_dir / "last.pt", model, cfg, epoch, {"train_loss": row["train_loss"]}, resume)
            if time_limit_reached:
                print(f"[train] time limit reached after {training_seconds / 60:.2f} training minutes")
                break
    finally:
        step_f.close()
        epoch_f.close()

    stats = log_run_stats("end", model, timer, vram)
    stats["training_seconds"] = round(training_seconds, 2)
    stats["training_minutes"] = round(training_seconds / 60, 2)
    stats["time_limit_reached"] = time_limit_reached
    stats.update(getattr(model, "adaptation_stats", {}))
    if start_epoch > 1:
        # wall_s / gpu_hours / peak_vram_gb above cover this session only
        stats["resumed_from_epoch"] = start_epoch - 1
        stats["wall_s_all_sessions"] = round(prior_wall_s + timer.elapsed(), 2)
        stats["gpu_hours_all_sessions"] = gpu_hours(prior_wall_s + timer.elapsed())
        stats["peak_vram_gb_all_sessions"] = max(prior_peak_vram_gb, vram.peak_gb())
    with (run_dir / "run_stats.txt").open("w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}={v}\n")
    print(f"[train] done. checkpoints in {run_dir}")
    return run_dir


def main():
    args = parse_args()
    if args.resume:
        # The run's own config, so the continued run trains exactly as it started.
        # Only the data location may change (a new Colab session).
        cfg = load_config(Path(args.resume) / "config.yaml")
        apply_overrides(cfg, data_root=args.data_root)
        train(cfg, resume_dir=args.resume)
        return
    cfg = load_config(args.config)
    apply_overrides(
        cfg,
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    train(cfg)


if __name__ == "__main__":
    main()
