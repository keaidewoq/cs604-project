"""Full fine-tuning loop with step checkpoints and a metrics log.

Checkpoints are written often enough that a Colab disconnect or a laptop sleep
can resume the same run. The metrics CSV is the loss curve for the report.
"""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cs604.data.dataset import CS604ImageDataset
from cs604.models.build import build_vit, parameter_report


class SyntheticImages(Dataset):
    """Fixed random tensors so the GPU path can be tested before images are extracted."""

    def __init__(self, length: int, num_classes: int, image_size: int):
        self.length = length
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        generator = torch.Generator()
        generator.manual_seed(index)
        image = torch.randn(3, self.image_size, self.image_size, generator=generator)
        label = index % self.num_classes
        return image, label, f"synthetic/{index:06d}.jpg"


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_loader(cfg: dict, train_transform) -> DataLoader:
    if cfg.get("synthetic"):
        dataset: Dataset = SyntheticImages(
            length=int(cfg.get("limit_samples") or 64),
            num_classes=int(cfg["num_classes"]),
            image_size=int(cfg["image_size"]),
        )
    else:
        dataset = CS604ImageDataset(cfg["data_root"], "train", transform=train_transform)
        limit = cfg.get("limit_samples")
        if limit is not None and int(limit) < len(dataset):
            rng = random.Random(int(cfg.get("seed", 0)))
            dataset.samples = rng.sample(dataset.samples, int(limit))

    workers = int(cfg.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _save(path: Path, model, optimizer, step: int, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )


def run_training(cfg: dict) -> Path:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. On this desktop, use the cu130 virtualenv "
            r"(C:\Users\Van Le\.venvs\cs604). On Colab, switch the runtime to GPU."
        )

    _seed(int(cfg.get("seed", 0)))
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_vit(
        checkpoint=cfg["timm_checkpoint"],
        num_classes=int(cfg["num_classes"]),
        grad_checkpointing=bool(cfg.get("grad_checkpointing", False)),
    )
    model.to(device)
    report = parameter_report(model)

    from timm.data import create_transform, resolve_model_data_config

    transform = None if cfg.get("synthetic") else create_transform(
        **resolve_model_data_config(model),
        is_training=True,
    )
    loader = _build_loader(cfg, transform)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.05)),
    )
    start_step = 0
    resume = cfg.get("resume")
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])

    precision = str(cfg.get("precision", "bf16"))
    use_amp = precision == "bf16"
    autocast_dtype = torch.bfloat16 if use_amp else torch.float32
    grad_accum = max(1, int(cfg.get("grad_accum", 1)))
    max_steps = cfg.get("max_steps")
    max_steps = None if max_steps is None else int(max_steps)
    log_every = int(cfg.get("log_every", 10))
    ckpt_every = int(cfg.get("ckpt_every", 100))
    epochs = int(cfg.get("epochs", 1))

    props = torch.cuda.get_device_properties(0)
    hardware = {
        "device": torch.cuda.get_device_name(0),
        "vram_gb": round(props.total_memory / (1024**3), 2),
        "precision": precision,
        "batch_size": int(cfg["batch_size"]),
        "grad_accum": grad_accum,
        **report,
    }
    (output_dir / "hardware.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")
    print(
        f"device={hardware['device']} vram={hardware['vram_gb']}GB "
        f"trainable={report['trainable_params']:,} ({report['trainable_pct']:.2f}%)"
    )

    metrics_path = output_dir / "metrics.csv"
    metrics = metrics_path.open("a" if start_step else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        metrics,
        fieldnames=["step", "loss", "images_per_sec", "max_mem_gb", "elapsed_sec"],
    )
    if start_step == 0:
        writer.writeheader()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    micro = 0
    images_since_log = 0
    log_t0 = time.perf_counter()
    run_t0 = log_t0
    window_loss = 0.0
    window_steps = 0
    micro_loss = 0.0

    for epoch in range(epochs):
        for images, labels, _paths in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
                loss = F.cross_entropy(model(images), labels)
            (loss / grad_accum).backward()
            micro_loss += float(loss.detach())
            micro += 1
            images_since_log += images.size(0)
            if micro % grad_accum != 0:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            window_loss += micro_loss / grad_accum
            window_steps += 1
            micro_loss = 0.0

            done = max_steps is not None and step >= max_steps
            if step % log_every == 0 or done:
                now = time.perf_counter()
                elapsed = now - log_t0
                row = {
                    "step": step,
                    "loss": window_loss / window_steps,
                    "images_per_sec": images_since_log / elapsed if elapsed else 0.0,
                    "max_mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
                    "elapsed_sec": now - run_t0,
                }
                writer.writerow(row)
                metrics.flush()
                print(
                    f"step {step} loss {row['loss']:.4f} "
                    f"{row['images_per_sec']:.1f} img/s "
                    f"mem {row['max_mem_gb']:.2f} GB"
                )
                window_loss = 0.0
                window_steps = 0
                images_since_log = 0
                log_t0 = now

            if step % ckpt_every == 0:
                _save(output_dir / "last.pt", model, optimizer, step, cfg)

            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    _save(output_dir / "last.pt", model, optimizer, step, cfg)
    hardware["elapsed_sec"] = time.perf_counter() - run_t0
    hardware["optimizer_steps"] = step
    (output_dir / "hardware.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")
    metrics.close()
    print(f"saved {output_dir / 'last.pt'} ({hardware['elapsed_sec']:.1f}s)")
    return output_dir / "last.pt"
