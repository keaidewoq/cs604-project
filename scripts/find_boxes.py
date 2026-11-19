"""Write an organism box for every training photo, for box-aware training crops (`train_box_file`).

How the score maps and boxes are made is in src/boxes.py. The default variant is DDT class
co-localization on the final tokens (what the first pilot used); --score / --layer / --mask /
--smooth pick another one, e.g. the winner of scripts/box_sweep.py. Nothing is trained, and
only <data_root>/train.txt is used (the pilot's fit images, never the holdout).

Also measures, for the report:
  - how often today's random training crops (RandomResizedCrop, scale 0.08-1 and 0.25-1)
    cover little of the box;
  - whether the boxes find the organism: class-prototype top-1 on the box crop vs a random
    crop of the same size vs the whole photo (`--check_per_class` photos per class);
  - an overlay figure of score maps and boxes.

    python scripts/find_boxes.py --config configs/lora_r24_336px_qkv-proj-fc_no-extras_proto-cosine_2.4h.yaml \\
        --data_root /content/pilot_data/c1000_h5_s0 --out_dir <folder> [--score proto --smooth ...]
writes <out_name>.pt, <out_name>_stats.json, <out_name>_overlay.png (out_name: boxes).
GPU time is printed and saved: report it as data preparation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.boxes import (FeatureTaps, WholeImages, box_from_map, class_scores, prototype_top1,
                       read_train_rows, variant_name)
from src.model import build_model
from src.utils import apply_overrides, get_device, load_config, set_seed


def sample_rrc(w, h, scale, ratio, rng):
    """timm/torchvision RandomResizedCrop geometry: (x0, y0, cw, ch) in pixels."""
    area = w * h
    log_r = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target = area * rng.uniform(*scale)
        r = math.exp(rng.uniform(*log_r))
        cw, ch = int(round(math.sqrt(target * r))), int(round(math.sqrt(target / r)))
        if 0 < cw <= w and 0 < ch <= h:
            return rng.randint(0, w - cw), rng.randint(0, h - ch), cw, ch
    in_r = w / h
    if in_r < ratio[0]:
        cw, ch = w, int(round(w / ratio[0]))
    elif in_r > ratio[1]:
        ch, cw = h, int(round(h * ratio[1]))
    else:
        cw, ch = w, h
    return (w - cw) // 2, (h - ch) // 2, cw, ch


def coverage(crop, box_px):
    x0, y0, cw, ch = crop
    bx0, by0, bx1, by1 = box_px
    ix = max(0.0, min(x0 + cw, bx1) - max(x0, bx0))
    iy = max(0.0, min(y0 + ch, by1) - max(y0, by0))
    return ix * iy / max((bx1 - bx0) * (by1 - by0), 1e-9)


def add_variant_args(p):
    p.add_argument("--score", choices=["ddt", "proto"], default="ddt")
    p.add_argument("--layer", type=int, default=None, help="block index (0-based); default: final tokens")
    p.add_argument("--mask", action="store_true", help="leave out high-norm outlier tokens")
    p.add_argument("--smooth", action="store_true", help="3x3 average of the score map")
    p.add_argument("--top", type=float, default=None, help="keep the photo's top fraction of patches (e.g. 0.2)")
    p.add_argument("--outlier_ratio", type=float, default=2.0, help="outlier: norm > ratio x median")
    p.add_argument("--margin", type=float, default=0.1, help="added on each side, fraction of the box size")
    p.add_argument("--min_patches", type=int, default=4)
    p.add_argument("--max_area", type=float, default=0.8, help="larger boxes are dropped (usual crops)")


def load_backbone(config):
    cfg = load_config(config)
    apply_overrides(cfg, method="linear_probe", head_init="random", pretrained=cfg.get("pretrained", True))
    set_seed(int(cfg.seed))
    device = get_device()
    model = build_model(cfg).to(device).eval()  # pretrained backbone; nothing is trained
    mean = model.pretrained_cfg.get("mean", (0.5,) * 3)
    std = model.pretrained_cfg.get("std", (0.5,) * 3)
    return cfg, device, model, mean, std


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--out_name", default="boxes")
    add_variant_args(p)
    p.add_argument("--check_per_class", type=int, default=5)
    p.add_argument("--crops_per_image", type=int, default=10, help="random crops sampled for the miss rate")
    p.add_argument("--overlays", type=int, default=24)
    args = p.parse_args()
    variant = {"score": args.score, "layer": args.layer, "mask": args.mask, "smooth": args.smooth, "top": args.top}
    variant["name"] = variant_name(variant)

    started = time.time()
    cfg, device, model, mean, std = load_backbone(args.config)
    size = int(cfg.img_size)
    grid = size // int(model.patch_embed.patch_size[0])
    use_bf16 = bool(cfg.bf16) and device.type == "cuda"
    taps = FeatureTaps(model, [args.layer])

    rows = read_train_rows(args.data_root)
    n_classes = len({r[1] for r in rows})
    print(f"[boxes] {variant['name']}: {len(rows)} training images, {n_classes} classes, grid {grid}x{grid}")

    loader = DataLoader(WholeImages(args.data_root, rows, size, mean, std), batch_size=int(cfg.batch_size),
                        shuffle=False, num_workers=int(cfg.num_workers), pin_memory=device.type == "cuda")
    boxes, sizes, maps_keep, prototypes, masked = [None] * len(rows), [None] * len(rows), {}, {}, []
    keep_maps = set(range(0, len(rows), max(1, len(rows) // max(args.overlays, 1))))
    buf = {"label": None, "idx": [], "feats": []}

    def flush():
        if not buf["idx"]:
            return
        feats = {k: torch.cat([f[k] for f in buf["feats"]]) for k in buf["feats"][0]}
        prototypes[buf["label"]] = F.normalize(feats["cls"].mean(0), dim=0)
        scores, frac = class_scores(feats, variant, grid, prototypes[buf["label"]], args.outlier_ratio)
        masked.append(frac)
        for k, i in enumerate(buf["idx"]):
            boxes[i] = box_from_map(scores[k], grid, args.margin, args.min_patches, args.max_area, args.top)
            if i in keep_maps:
                maps_keep[i] = scores[k].view(grid, grid).clone()
        buf.update(idx=[], feats=[])

    for images, labels, idx, wh in tqdm(loader, desc="boxes"):
        out = taps(images.to(device, non_blocking=True), use_bf16)
        for k in range(len(labels)):
            label, i = int(labels[k]), int(idx[k])
            if buf["label"] is not None and label != buf["label"]:
                flush()
            buf["label"] = label
            buf["idx"].append(i)
            buf["feats"].append({key: v[k:k + 1].half() if v.dim() == 3 else v[k:k + 1] for key, v in out.items()})
            sizes[i] = tuple(int(v) for v in wh[k])
    flush()
    box_seconds = time.time() - started

    # how often today's random crops show little of the box
    has_box = [i for i in range(len(rows)) if boxes[i] is not None]
    rng = random.Random(0)
    miss = {}
    for name, scale in (("rrc_scale_0.08", (0.08, 1.0)), ("rrc_scale_0.25", (0.25, 1.0))):
        cov = []
        for i in has_box:
            (w, h), b = sizes[i], boxes[i]
            box_px = (b[0] * w, b[1] * h, b[2] * w, b[3] * h)
            cov += [coverage(sample_rrc(w, h, scale, (3 / 4, 4 / 3), rng), box_px)
                    for _ in range(args.crops_per_image)]
        cov = torch.tensor(cov or [1.0])
        miss[name] = {"crops_with_under_50pct_of_box": round(float((cov < 0.5).float().mean()), 4),
                      "crops_with_under_10pct_of_box": round(float((cov < 0.1).float().mean()), 4),
                      "median_box_coverage": round(float(cov.median()), 4)}
    box_area = torch.tensor([(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) for i in has_box] or [1.0])

    # do the boxes find the organism? prototype top-1 on box vs same-size random crop
    per_class, items = {}, {"box": [], "random": [], "whole": []}
    for i in has_box:
        label = rows[i][1]
        if per_class.get(label, 0) >= args.check_per_class:
            continue
        per_class[label] = per_class.get(label, 0) + 1
        b = boxes[i]
        bw, bh = b[2] - b[0], b[3] - b[1]
        rx, ry = rng.uniform(0, 1 - bw), rng.uniform(0, 1 - bh)
        items["box"].append((i, b))
        items["random"].append((i, [rx, ry, rx + bw, ry + bh]))
        items["whole"].append((i, [0.0, 0.0, 1.0, 1.0]))
    check = {f"top1_{name}_crop": round(prototype_top1(model, args.data_root, rows, its, prototypes, size, mean, std,
                                                       int(cfg.batch_size), int(cfg.num_workers), device, use_bf16), 4)
             for name, its in items.items()}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"boxes": {rows[i][0]: boxes[i] for i in has_box}, "variant": variant,
                "img_size": size, "margin": args.margin}, out / f"{args.out_name}.pt")
    stats = {
        "variant": variant,
        "images": len(rows), "classes": n_classes,
        "images_with_box": len(has_box),
        "images_without_box_usual_crops": len(rows) - len(has_box),
        "outlier_tokens_masked": round(sum(masked) / max(len(masked), 1), 4) if args.mask else None,
        "box_area_fraction": {"mean": round(float(box_area.mean()), 4),
                              "median": round(float(box_area.median()), 4)},
        "random_crop_miss": miss,
        "box_check_prototype_top1": check,
        "box_finding_seconds": round(box_seconds, 1),
        "total_seconds": round(time.time() - started, 1),
        "device": str(device),
    }
    (out / f"{args.out_name}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    if maps_keep:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        shown = sorted(maps_keep)[:args.overlays]
        cols = 6
        fig, axes = plt.subplots(math.ceil(len(shown) / cols), cols, figsize=(3 * cols, 3 * math.ceil(len(shown) / cols)),
                                 squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, i in zip(axes.flat, shown):
            image = Image.open(Path(args.data_root) / rows[i][0]).convert("RGB").resize((size, size))
            ax.imshow(image)
            heat = maps_keep[i].clamp_min(0)
            ax.imshow(F.interpolate(heat[None, None], size=(size, size), mode="bilinear")[0, 0],
                      alpha=0.45, cmap="jet", extent=(0, size, size, 0))
            b = boxes[i] or [0.0, 0.0, 0.0, 0.0]
            ax.add_patch(Rectangle((b[0] * size, b[1] * size), (b[2] - b[0]) * size, (b[3] - b[1]) * size,
                                   fill=False, edgecolor="lime", linewidth=2))
            ax.set_title(f"class {rows[i][1]}" + ("" if boxes[i] else " (no box)"), fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{args.out_name}_overlay.png", dpi=110)
        print(f"[boxes] overlay -> {out / f'{args.out_name}_overlay.png'}")
    print(f"[boxes] {len(has_box)} of {len(rows)} images got a box -> {out / f'{args.out_name}.pt'}; "
          f"box finding took {box_seconds / 60:.1f} min (data preparation, report separately)")


if __name__ == "__main__":
    main()
