"""Image-list dataset and dataloaders.

Label files are one `image_path class_id` per line. `val.txt` / `test.txt`
may omit the class id; those splits are then unlabeled.
"""

from __future__ import annotations

from pathlib import Path
import math
import random

from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from torchvision.transforms import functional as TF
from timm.data import create_transform, resolve_data_config
from timm.models import get_pretrained_cfg


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ResizeWithPadding:
    """Keep the entire image and its proportions, then center-pad to a square."""

    def __init__(self, size, fill):
        self.size = int(size)
        self.fill = tuple(fill)

    def __call__(self, image):
        scale = self.size / max(image.size)
        width = max(1, min(self.size, round(image.width * scale)))
        height = max(1, min(self.size, round(image.height * scale)))
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        left, top = (self.size - width) // 2, (self.size - height) // 2
        return ImageOps.expand(
            image, (left, top, self.size - width - left, self.size - height - top),
            fill=self.fill,
        )


class RandomCropOrPad:
    """Choose exactly one view; subsequent flip/color/normalization are shared."""

    def __init__(self, crop, pad, probability):
        self.crop = crop
        self.pad = pad
        self.probability = probability

    def __call__(self, image):
        return self.pad(image) if random.random() < self.probability else self.crop(image)

    def __repr__(self):
        return f"RandomCropOrPad(full_image_probability={self.probability}, crop={self.crop})"


class BoxAwareRandomResizedCrop:
    """RandomResizedCrop that always keeps the object box inside the crop (`train_box_file`).

    Boxes ([x0, y0, x1, y1] as fractions of the image, by image path) come from
    scripts/find_boxes.py. Size and aspect are drawn as in `rrc` (its scale and
    ratio), then grown to at least the box and placed at random wherever the box
    still fits, so every view shows the whole object with a random amount of
    surroundings. Images without a box use `rrc` itself.
    """

    def __init__(self, rrc, boxes):
        self.rrc = rrc
        self.boxes = boxes
        self.size = rrc.size
        self.scale = rrc.scale
        self.ratio = rrc.ratio
        self.interpolation = rrc.interpolation

    def __call__(self, image, rel_path):
        box = self.boxes.get(rel_path)
        if box is None:
            return self.rrc(image)
        w, h = image.size
        bx0, by0, bx1, by1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
        bw, bh = max(bx1 - bx0, 1.0), max(by1 - by0, 1.0)
        log_r = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            area = w * h * random.uniform(*self.scale)
            r = math.exp(random.uniform(*log_r))
            cw = min(max(math.sqrt(area * r), bw), w)
            ch = min(max(math.sqrt(area / r), bh), h)
            if cw / ch < self.ratio[0] * 0.5 or cw / ch > self.ratio[1] * 2:
                continue  # growing to the box made it far too thin
            x0 = random.uniform(max(0.0, bx1 - cw), min(bx0, w - cw))
            y0 = random.uniform(max(0.0, by1 - ch), min(by0, h - ch))
            break
        else:
            x0, y0, cw, ch = 0.0, 0.0, float(w), float(h)
        # round outwards so the whole box stays inside the crop
        left, top = math.floor(x0), math.floor(y0)
        right, bottom = min(w, math.ceil(x0 + cw)), min(h, math.ceil(y0 + ch))
        return TF.resized_crop(image, top, left, max(1, bottom - top), max(1, right - left),
                               list(self.size), self.interpolation)

    def __repr__(self):
        return f"BoxAwareRandomResizedCrop({len(self.boxes)} boxes, scale={self.scale}, ratio={self.ratio})"


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
    def __init__(self, data_root, split, transform=None, box_crop=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.box_crop = box_crop  # BoxAwareRandomResizedCrop, applied before `transform`
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
        if self.box_crop is not None:
            image = self.box_crop(image, rel_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label, rel_path


def build_transforms(cfg, eval_crop_pct=None, img_size=None):
    """Train/eval transforms from the chosen timm model's data config.

    ``eval_crop_pct`` overrides the eval center-crop fraction (default: the
    model's own, 0.9 for the team backbone); used by ``src.predict --crop_pct``.
    ``img_size`` overrides ``cfg.img_size`` (the low-res phase of progressive resizing).
    """
    size = int(img_size or cfg.img_size)
    pretrained_cfg = get_pretrained_cfg(cfg.model_name)
    if pretrained_cfg is not None and hasattr(pretrained_cfg, "to_dict"):
        pretrained_cfg = pretrained_cfg.to_dict()
    data_config = resolve_data_config(
        {"input_size": (3, size, size)},
        pretrained_cfg=pretrained_cfg,
    )
    crop_scale = cfg.get("train_crop_scale")
    if crop_scale is not None:
        crop_scale = tuple(float(v) for v in crop_scale)
        if len(crop_scale) != 2 or not 0 < crop_scale[0] <= crop_scale[1] <= 1:
            raise ValueError(f"train_crop_scale must be [min, max] with 0 < min <= max <= 1, got {crop_scale}")
    # RandomResizedCrop area range; timm's default is (0.08, 1.0)
    train_tf = create_transform(**data_config, is_training=True,
                                **({"scale": crop_scale} if crop_scale else {}))
    probability = float(getattr(cfg, "full_image_probability", 0.0))
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("full_image_probability must be between 0 and 1")
    if probability:
        # Replace only geometry. Every view still receives the existing flip,
        # color jitter, tensor conversion and pretrained normalization.
        fill = tuple(round(255 * value) for value in data_config["mean"])
        train_tf.transforms[0] = RandomCropOrPad(
            train_tf.transforms[0], ResizeWithPadding(size, fill), probability,
        )
    if eval_crop_pct is None:
        eval_crop_pct = getattr(cfg, "eval_crop_pct", None)
    if eval_crop_pct is not None:
        data_config = {**data_config, "crop_pct": float(eval_crop_pct)}
    eval_tf = create_transform(**data_config, is_training=False)
    return train_tf, eval_tf


def _make_loader(dataset, cfg, shuffle, drop_last, persistent=True):
    pin = bool(getattr(cfg, "pin_memory", True))
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=drop_last,
        persistent_workers=persistent and cfg.num_workers > 0,
    )


def build_dataloaders(cfg):
    train_tf, eval_tf = build_transforms(cfg)
    root = Path(cfg.data_root)

    box_crop = None
    if cfg.get("train_box_file"):
        if cfg.get("low_res_size") or float(cfg.get("full_image_probability", 0.0)):
            raise ValueError("train_box_file cannot be combined with low_res_size or full_image_probability")
        import torch

        boxes = torch.load(cfg.train_box_file, map_location="cpu", weights_only=False)["boxes"]
        box_crop = BoxAwareRandomResizedCrop(train_tf.transforms[0], boxes)
        train_tf = Compose(train_tf.transforms[1:])  # the box-aware crop replaces the random crop
        print(f"[data] box-aware training crops: {len(boxes):,} boxes from {cfg.train_box_file}")
    train_ds = ImageListDataset(root, "train", transform=train_tf, box_crop=box_crop)
    loaders = {
        "train": _make_loader(train_ds, cfg, shuffle=True, drop_last=True),
        "val": None,
        "test": None,
        "has_val_labels": False,
        "has_test_labels": False,
        "train_low": None,
    }
    # Progressive resizing: the same training images at low_res_size, for the
    # first low_res_fraction of training (src/train.py switches loaders).
    if cfg.get("low_res_size"):
        low_tf, _ = build_transforms(cfg, img_size=int(cfg.low_res_size))
        low_ds = ImageListDataset(root, "train", transform=low_tf)
        # not persistent: its workers exit once the low-res phase is over
        loaders["train_low"] = _make_loader(low_ds, cfg, shuffle=True, drop_last=True, persistent=False)

    for split in ("val", "test"):
        list_file = root / f"{split}.txt"
        folder = root / split
        if not list_file.exists() and not folder.is_dir():
            continue
        ds = ImageListDataset(root, split, transform=eval_tf)
        loaders[split] = _make_loader(ds, cfg, shuffle=False, drop_last=False)
        loaders[f"has_{split}_labels"] = ds.has_labels

    return loaders
