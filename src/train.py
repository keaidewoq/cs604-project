"""Train entry point: python -m src.train --config configs/X.yaml"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.data import build_dataloaders
from src.losses import build_loss
from src.model import build_model
from src.utils import (
    PEFT_METHODS,
    Timer,
    VRAMTracker,
    apply_overrides,
    get_device,
    load_config,
    log_run_stats,
    save_config,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune a ViT with a method config.")
    p.add_argument("--config", required=True, help="Path to a method YAML (e.g. configs/lora_r8.yaml)")
    p.add_argument("--data_root", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    return p.parse_args()


def adamw_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
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


def save_checkpoint(path, model, cfg, epoch, metrics):
    payload = {
        "state_dict": collect_state(model, cfg.method),
        "epoch": epoch,
        "metrics": metrics or {},
        "method": cfg.method,
        "model_name": cfg.model_name,
        "num_classes": cfg.num_classes,
        "trainable_only": cfg.method in PEFT_METHODS,
    }
    torch.save(payload, path)


def amp_context(device, use_bf16):
    enabled = bool(use_bf16) and device.type == "cuda"
    dtype = torch.bfloat16 if enabled else torch.float32
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_bf16):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_context(device, use_bf16):
            logits = model(images)
            loss = loss_fn(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)
    return {"val_loss": total_loss / max(n, 1), "val_acc": correct / max(n, 1)}


def train(cfg):
    set_seed(int(cfg.seed))
    device = get_device()
    print(f"[train] device={device} method={cfg.method} model={cfg.model_name}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir) / f"{cfg.method}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    print(f"[train] run_dir={run_dir}")

    timer = Timer()
    vram = VRAMTracker()

    loaders = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)

    groups = adamw_param_groups(model, float(cfg.weight_decay))
    optimizer = AdamW(groups, lr=float(cfg.lr))

    accum = max(int(cfg.grad_accum_steps), 1)
    steps_per_epoch = math.ceil(len(loaders["train"]) / accum)
    total_steps = max(steps_per_epoch * int(cfg.epochs), 1)
    warmup_steps = int(float(cfg.warmup) * steps_per_epoch)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    log_run_stats("start", model, timer, vram)
    vram.reset()

    step_path = run_dir / "step_log.csv"
    epoch_path = run_dir / "epoch_log.csv"
    step_f = step_path.open("w", newline="", encoding="utf-8")
    epoch_f = epoch_path.open("w", newline="", encoding="utf-8")
    step_w = csv.DictWriter(step_f, fieldnames=["step", "epoch", "loss", "lr"])
    epoch_w = csv.DictWriter(
        epoch_f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "epoch_time_s"]
    )
    step_w.writeheader()
    epoch_w.writeheader()

    best_acc = -1.0
    global_step = 0
    use_bf16 = bool(cfg.bf16)
    has_val = bool(loaders["val"] is not None and loaders["has_val_labels"])

    try:
        for epoch in range(1, int(cfg.epochs) + 1):
            model.train()
            epoch_timer = Timer()
            running = 0.0
            seen = 0
            optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(loaders["train"], desc=f"epoch {epoch}/{cfg.epochs}", leave=True)
            n_batches = len(loaders["train"])
            for i, (images, labels, _) in enumerate(pbar, start=1):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with amp_context(device, use_bf16):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                (loss / accum).backward()

                if i % accum == 0 or i == n_batches:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                loss_val = float(loss.item())
                running += loss_val * labels.size(0)
                seen += labels.size(0)
                lr = scheduler.get_last_lr()[0]
                step_w.writerow(
                    {"step": global_step, "epoch": epoch, "loss": loss_val, "lr": lr}
                )
                pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}")

            step_f.flush()

            row = {
                "epoch": epoch,
                "train_loss": running / max(seen, 1),
                "val_loss": "",
                "val_acc": "",
                "epoch_time_s": round(epoch_timer.elapsed(), 2),
            }
            if has_val:
                metrics = evaluate(model, loaders["val"], loss_fn, device, use_bf16)
                row["val_loss"] = metrics["val_loss"]
                row["val_acc"] = metrics["val_acc"]
                print(
                    f"[epoch {epoch}] train_loss={row['train_loss']:.4f} "
                    f"val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_acc']:.4f}"
                )
                if metrics["val_acc"] > best_acc:
                    best_acc = metrics["val_acc"]
                    save_checkpoint(run_dir / "best.pt", model, cfg, epoch, metrics)
                    print(f"[epoch {epoch}] new best val_acc={best_acc:.4f}")
            else:
                print(f"[epoch {epoch}] train_loss={row['train_loss']:.4f} (no labeled val)")

            epoch_w.writerow(row)
            epoch_f.flush()
            save_checkpoint(run_dir / "last.pt", model, cfg, epoch, {"train_loss": row["train_loss"]})
    finally:
        step_f.close()
        epoch_f.close()

    stats = log_run_stats("end", model, timer, vram)
    with (run_dir / "run_stats.txt").open("w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}={v}\n")
    print(f"[train] done. checkpoints in {run_dir}")
    return run_dir


def main():
    args = parse_args()
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
