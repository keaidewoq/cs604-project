"""Image-list dataset and dataloaders.

Label files are one `image_path class_id` per line. `val.txt` / `test.txt`
may omit the class id; those splits are then unlabeled.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from timm.data import create_transform, resolve_data_config
from timm.models import get_pretrained_cfg


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _parse_line(line):
    parts = line.strip().split()
    if not parts:
        return None
    path = parts[0]
    label = int(parts[1]) if len(parts) >= 2 else -1
    return path, label


def _scan_folder(data_root, split):
    folder = Path(data_root) / split
    if not folder.is_dir():
        return []
    samples = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTS:
            rel = path.relative_to(data_root).as_posix()
            samples.append((rel, -1))
    return samples


class ImageListDataset(Dataset):
    def __init__(self, data_root, split, transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()
        self.has_labels = bool(self.samples) and all(y >= 0 for _, y in self.samples)

    def _load_samples(self):
        list_file = self.data_root / f"{self.split}.txt"
        if list_file.exists():
            samples = []
            with list_file.open("r", encoding="utf-8") as f:
                for line in f:
                    parsed = _parse_line(line)
                    if parsed is not None:
                        samples.append(parsed)
            return samples
        scanned = _scan_folder(self.data_root, self.split)
        if scanned:
            print(f"[data] {list_file.name} missing; scanned {self.split}/ ({len(scanned)} images)")
            return scanned
        raise FileNotFoundError(
            f"No {list_file} and no {self.data_root / self.split} folder under {self.data_root}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        image = Image.open(self.data_root / rel_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, rel_path


def build_transforms(cfg, eval_crop_pct=None):
    """Train/eval transforms from the chosen timm model's data config.

    ``eval_crop_pct`` overrides the eval center-crop fraction (default: the
    model's own, 0.9 for the team backbone); used by ``src.predict --crop_pct``.
    """
    pretrained_cfg = get_pretrained_cfg(cfg.model_name)
    if pretrained_cfg is not None and hasattr(pretrained_cfg, "to_dict"):
        pretrained_cfg = pretrained_cfg.to_dict()
    data_config = resolve_data_config(
        {"input_size": (3, cfg.img_size, cfg.img_size)},
        pretrained_cfg=pretrained_cfg,
    )
    train_tf = create_transform(**data_config, is_training=True)
    if eval_crop_pct is not None:
        data_config = {**data_config, "crop_pct": float(eval_crop_pct)}
    eval_tf = create_transform(**data_config, is_training=False)
    return train_tf, eval_tf


def _make_loader(dataset, cfg, shuffle, drop_last):
    pin = bool(getattr(cfg, "pin_memory", True))
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=drop_last,
        persistent_workers=cfg.num_workers > 0,
    )


def build_dataloaders(cfg):
    train_tf, eval_tf = build_transforms(cfg)
    root = Path(cfg.data_root)

    train_ds = ImageListDataset(root, "train", transform=train_tf)
    loaders = {
        "train": _make_loader(train_ds, cfg, shuffle=True, drop_last=True),
        "val": None,
        "test": None,
        "has_val_labels": False,
        "has_test_labels": False,
    }

    for split in ("val", "test"):
        list_file = root / f"{split}.txt"
        folder = root / split
        if not list_file.exists() and not folder.is_dir():
            continue
        ds = ImageListDataset(root, split, transform=eval_tf)
        loaders[split] = _make_loader(ds, cfg, shuffle=False, drop_last=False)
        loaders[f"has_{split}_labels"] = ds.has_labels

    return loaders
