"""Checkpoint inference and strict CodaBench submission validation."""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cs604.data.dataset import CS604ImageDataset, load_split_index
from cs604.models.build import build_from_config


def validate_submission(
    submission_path: str | Path,
    expected_index: str | Path,
    num_classes: int = 10_000,
) -> None:
    expected = [path for path, _label in load_split_index(Path(expected_index))]
    lines = Path(submission_path).read_text(encoding="utf-8").splitlines()
    if len(lines) != len(expected):
        raise ValueError(f"Expected {len(expected)} predictions, found {len(lines)}")
    for number, (line, expected_path) in enumerate(zip(lines, expected), start=1):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Line {number} must contain image_path and class_id")
        rel_path, raw_label = parts
        if rel_path != expected_path:
            raise ValueError(
                f"Line {number} path is {rel_path!r}; expected {expected_path!r}"
            )
        try:
            label = int(raw_label)
        except ValueError as error:
            raise ValueError(f"Line {number} has a non-integer class ID") from error
        if not 0 <= label < num_classes:
            raise ValueError(f"Line {number} class ID {label} is outside the valid range")


@torch.inference_mode()
def predict_split(
    checkpoint_path: str | Path,
    data_root: str | Path,
    split: str,
    output_path: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    make_zip: bool = True,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for prediction")
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = build_from_config(cfg)
    model.load_state_dict(checkpoint["model"])
    model.cuda().eval()

    from timm.data import create_transform, resolve_model_data_config

    transform = create_transform(
        **resolve_model_data_config(model), is_training=False
    )
    dataset = CS604ImageDataset(data_root, split, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    precision = str(cfg.get("precision", "bf16"))
    amp_dtype = torch.bfloat16 if precision == "bf16" else (
        torch.float16 if precision == "fp16" else None
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for images, _labels, paths in loader:
            images = images.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                predictions = model(images).argmax(dim=1).cpu().tolist()
            for rel_path, label in zip(paths, predictions):
                handle.write(f"{rel_path} {label}\n")
            count += len(predictions)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    index_path = Path(data_root) / f"{split}.txt"
    validate_submission(output_path, index_path, int(cfg["num_classes"]))
    metrics = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "samples": count,
        "elapsed_sec": elapsed,
        "images_per_sec": count / max(elapsed, 1e-9),
        "milliseconds_per_image": 1000 * elapsed / max(count, 1),
        "device": torch.cuda.get_device_name(0),
    }
    output_path.with_suffix(".inference.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if make_zip:
        zip_path = output_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(output_path, arcname=output_path.name)
    return output_path
