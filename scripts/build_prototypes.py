"""Average train-set features per class (Van's 43.9% zero-shot setup).

Needs the real dataset and a GPU on Colab. Writes a .pt the trainer can load
via `head_init: prototype` + `prototype_file`.

  python scripts/build_prototypes.py --config configs/prototype_probe.yaml \\
      --output /content/drive/MyDrive/CS604_RUNS/prototypes.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ImageListDataset, build_transforms
from src.model import build_model
from src.utils import apply_overrides, get_device, load_config


def _cap_per_class(samples, k, seed):
    if k is None:
        return samples
    by_class = defaultdict(list)
    for path, label in samples:
        by_class[label].append((path, label))
    rng = torch.Generator().manual_seed(int(seed))
    out = []
    for items in by_class.values():
        perm = torch.randperm(len(items), generator=rng).tolist()
        out.extend(items[i] for i in perm[: int(k)])
    return out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prototype_probe.yaml")
    parser.add_argument("--output", required=True, help="Where to write prototypes.pt")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--per_class", type=int, default=None, help="Max train images per class")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--base_prototypes", default=None,
                        help="Existing prototypes.pt; classes with no images here keep their rows from it "
                             "(rebuild only a class subset, e.g. without a pilot holdout)")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    apply_overrides(cfg, data_root=args.data_root, pretrained=True)
    device = get_device()
    if device.type != "cuda":
        print("[prototypes] warning: no GPU; this will be very slow")

    _, eval_tf = build_transforms(cfg)
    ds = ImageListDataset(cfg.data_root, "train", transform=eval_tf)
    if not ds.has_labels:
        raise SystemExit("train.txt must have labels to build prototypes")
    per_class = args.per_class if args.per_class is not None else cfg.get("prototype_per_class")
    samples = _cap_per_class(ds.samples, per_class, cfg.get("seed", 42))
    ds.samples = samples
    ds.has_labels = True

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )

    # Frozen backbone, any method is fine; we only need features.
    apply_overrides(cfg, method="linear_probe", head_init="random")
    model = build_model(cfg).to(device).eval()
    feat_dim = model.num_features
    n_cls = int(cfg.num_classes)
    sums = torch.zeros(n_cls, feat_dim, device=device)
    counts = torch.zeros(n_cls, device=device)

    use_bf16 = bool(cfg.bf16) and device.type == "cuda"
    for images, labels, _ in tqdm(loader, desc="prototypes"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            feats = model.forward_head(model.forward_features(images), pre_logits=True)
        feats = F.normalize(feats.float(), dim=1)
        sums.index_add_(0, labels, feats)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))

    empty_mask = counts == 0
    empty = int(empty_mask.sum())
    if empty and not args.base_prototypes:
        raise RuntimeError(f"{empty} classes have no images; check train.txt (or pass --base_prototypes)")
    prototypes = F.normalize(sums / counts.clamp(min=1)[:, None], dim=1).cpu()
    if empty:
        base = torch.load(args.base_prototypes, map_location="cpu", weights_only=False)
        if isinstance(base, dict):
            built = base.get("model_name")
            if built and built != cfg.model_name:
                raise ValueError(f"Base prototypes were built with {built}, not {cfg.model_name}")
            base = base["prototypes"]
        if tuple(base.shape) != tuple(prototypes.shape):
            raise ValueError(f"Base prototypes shape {tuple(base.shape)} != {tuple(prototypes.shape)}")
        keep = empty_mask.cpu()
        prototypes[keep] = F.normalize(base[keep].float(), dim=1)
        print(f"[prototypes] rebuilt {n_cls - empty} classes; kept {empty} rows from {args.base_prototypes}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prototypes": prototypes,
            "samples_per_class": counts.cpu(),
            "model_name": cfg.model_name,
            "img_size": int(cfg.img_size),
        },
        out,
    )
    summary = {
        "model_name": cfg.model_name,
        "img_size": int(cfg.img_size),
        "train_images": len(samples),
        "classes": n_cls,
        "empty_classes": empty,
        "base_prototypes": args.base_prototypes,
        "output": str(out),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[prototypes] saved {out}  shape={tuple(prototypes.shape)}")


if __name__ == "__main__":
    main()
