"""Find the organism in every training image from what its class's images have in common.

Object co-localization with DDT (Deep Descriptor Transforming; Wei et al., IJCAI 2017 /
Pattern Recognition 2019): the images of one class share the object, while their
backgrounds differ. So, per class:

  1. every image goes through the PRETRAINED backbone (nothing is trained), whole image
     resized to img_size x img_size, and its final patch tokens are kept
     (21 x 21 = 441 patches of 16 px at 336);
  2. the patch features of all the class's images are centred and their first principal
     component is found: the direction most patches of this class vary along together,
     which across 40+ different photos is the shared object;
  3. each patch's projection on it is a score map. Its sign is arbitrary, so it is set
     with the boundary prior from saliency work: the object scores lower on the patches
     along the photo's edge than inside (how often the class prototype would have picked
     the other sign is reported);
  4. the box is the bounding rectangle of the largest connected blob of positive patches,
     plus a margin. With no blob of `--min_patches`, or a box over `--max_area` of the
     photo, the image gets no box and keeps the usual random crops in training.

Boxes are stored as fractions of the original image ([x0, y0, x1, y1] in 0..1), so the
training crop (src/data.py, `train_box_file`) can use them at any resolution.

Also measures, for the report:
  - how often today's random training crops (RandomResizedCrop, scale 0.08-1 and 0.25-1)
    cover little of the box, i.e. how often a training view shows little of the organism;
  - whether the boxes find the organism: class-prototype top-1 on the box crop vs a random
    crop of the same size vs the whole image (`--check_per_class` images per class);
  - an overlay figure of score maps and boxes.

    python scripts/find_boxes.py --config configs/lora_r24_336_no_extras.yaml \\
        --data_root /content/pilot_data/c1000_h5_s0 --out_dir <folder>
writes boxes.pt, boxes_stats.json, boxes_overlay.png. Uses the training images in
<data_root>/train.txt only. GPU time is printed and saved: report it as data preparation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import build_model
from src.utils import apply_overrides, get_device, load_config, set_seed


class WholeImages(Dataset):
    """Whole images squashed to size x size (no crop, so the patch grid covers the full photo)."""

    def __init__(self, root, rows, size, mean, std):
        self.root, self.rows = Path(root), rows
        self.tf = T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
                             T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i]
        image = Image.open(self.root / path).convert("RGB")
        return self.tf(image), label, i, torch.tensor(image.size)


class Crops(Dataset):
    """(image index, [x0, y0, x1, y1] fraction) -> that region resized to size x size."""

    def __init__(self, root, rows, items, size, mean, std):
        self.root, self.rows, self.items = Path(root), rows, items
        self.tf = T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
                             T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, box = self.items[k]
        image = Image.open(self.root / self.rows[i][0]).convert("RGB")
        w, h = image.size
        x0, y0, x1, y1 = box
        region = image.crop((int(x0 * w), int(y0 * h), max(int(x0 * w) + 1, round(x1 * w)),
                             max(int(y0 * h) + 1, round(y1 * h))))
        return self.tf(region), self.rows[i][1]


def largest_blob(mask):
    """Largest 4-connected True region of a (g, g) bool grid -> (rows, cols) or None."""
    g = mask.shape[0]
    seen = torch.zeros_like(mask)
    best = None
    for r in range(g):
        for c in range(g):
            if not mask[r, c] or seen[r, c]:
                continue
            blob, queue = [], deque([(r, c)])
            seen[r, c] = True
            while queue:
                y, x = queue.popleft()
                blob.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < g and 0 <= nx < g and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            if best is None or len(blob) > len(best):
                best = blob
    return best


def border_mask(grid):
    m = torch.zeros(grid, grid, dtype=torch.bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m.flatten()


def ddt_maps(patches, cls_feats, grid):
    """patches (n, P, C), cls_feats (n, C) of one class -> ((n, P) score maps, positive = object;
    whether the class-prototype sign rule would have disagreed)."""
    n, p, c = patches.shape
    x = patches.reshape(-1, c).double()
    x = x - x.mean(0, keepdim=True)
    cov = x.T @ x / x.shape[0]
    _, vecs = torch.linalg.eigh(cov)
    proj = (x @ vecs[:, -1]).float().view(n, p)
    # sign (boundary prior): photo-edge patches should score lower than the inside
    edge = border_mask(grid)
    if proj[:, edge].mean() > proj[:, ~edge].mean():
        proj = -proj
    # the alternative rule, for the record: agree with the class prototype's patch-similarity map
    proto = F.normalize(cls_feats.float().mean(0), dim=0)
    sim = F.normalize(patches.float(), dim=-1) @ proto
    sim = sim - sim.mean(1, keepdim=True)
    return proj, bool((proj * sim).sum() < 0)


def box_from_map(score, grid, margin, min_patches, max_area):
    """(P,) score map -> ([x0, y0, x1, y1] fractions, or None when no useful box)."""
    blob = largest_blob((score > 0).view(grid, grid))
    if blob is None or len(blob) < min_patches:
        return None
    ys, xs = [b[0] for b in blob], [b[1] for b in blob]
    x0, x1 = min(xs) / grid, (max(xs) + 1) / grid
    y0, y1 = min(ys) / grid, (max(ys) + 1) / grid
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    box = [max(0.0, x0 - mx), max(0.0, y0 - my), min(1.0, x1 + mx), min(1.0, y1 + my)]
    if (box[2] - box[0]) * (box[3] - box[1]) > max_area:
        return None  # nearly the whole photo: aiming adds nothing, keep the usual crops
    return box


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


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--margin", type=float, default=0.1, help="added on each side, fraction of the box size")
    p.add_argument("--min_patches", type=int, default=4)
    p.add_argument("--max_area", type=float, default=0.8, help="larger boxes are dropped (usual crops)")
    p.add_argument("--check_per_class", type=int, default=5)
    p.add_argument("--crops_per_image", type=int, default=10, help="random crops sampled for the miss rate")
    p.add_argument("--overlays", type=int, default=24)
    args = p.parse_args()

    started = time.time()
    cfg = load_config(args.config)
    apply_overrides(cfg, method="linear_probe", head_init="random", pretrained=cfg.get("pretrained", True))
    set_seed(int(cfg.seed))
    device = get_device()
    model = build_model(cfg).to(device).eval()  # pretrained backbone; nothing is trained
    size = int(cfg.img_size)
    grid = size // int(model.patch_embed.patch_size[0])
    prefix = int(getattr(model, "num_prefix_tokens", 1))
    mean, std = model.pretrained_cfg.get("mean", (0.5,) * 3), model.pretrained_cfg.get("std", (0.5,) * 3)
    use_bf16 = bool(cfg.bf16) and device.type == "cuda"

    rows = []
    for line in (Path(args.data_root) / "train.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rows.append((parts[0], int(parts[1])))
    order = sorted(range(len(rows)), key=lambda i: (rows[i][1], i))  # class by class
    rows = [rows[i] for i in order]
    n_classes = len({r[1] for r in rows})
    print(f"[boxes] {len(rows)} training images, {n_classes} classes, grid {grid}x{grid}")

    loader = DataLoader(WholeImages(args.data_root, rows, size, mean, std), batch_size=int(cfg.batch_size),
                        shuffle=False, num_workers=int(cfg.num_workers), pin_memory=device.type == "cuda")
    boxes, sizes, maps_keep = [None] * len(rows), [None] * len(rows), {}
    prototypes, sign_disagree = {}, 0
    buf = {"label": None, "idx": [], "patch": [], "cls": []}
    keep_maps = set(range(0, len(rows), max(1, len(rows) // max(args.overlays, 1))))

    def flush():
        nonlocal sign_disagree
        if not buf["idx"]:
            return
        patches, cls_feats = torch.cat(buf["patch"]), torch.cat(buf["cls"])
        prototypes[buf["label"]] = F.normalize(cls_feats.mean(0), dim=0)
        scores, disagree = ddt_maps(patches, cls_feats, grid)
        sign_disagree += disagree
        for k, i in enumerate(buf["idx"]):
            boxes[i] = box_from_map(scores[k], grid, args.margin, args.min_patches, args.max_area)
            if i in keep_maps:
                maps_keep[i] = scores[k].view(grid, grid).clone()
        buf.update(idx=[], patch=[], cls=[])

    for images, labels, idx, wh in tqdm(loader, desc="boxes"):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            tokens = model.forward_features(images.to(device, non_blocking=True))
            feats = model.forward_head(tokens, pre_logits=True)
        tokens, feats = tokens[:, prefix:].float().cpu(), feats.float().cpu()
        for k in range(len(labels)):
            label, i = int(labels[k]), int(idx[k])
            if buf["label"] is not None and label != buf["label"]:
                flush()
            buf["label"] = label
            buf["idx"].append(i)
            buf["patch"].append(tokens[k:k + 1].half())
            buf["cls"].append(feats[k:k + 1])
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
        cov = torch.tensor(cov)
        miss[name] = {"crops_with_under_50pct_of_box": round(float((cov < 0.5).float().mean()), 4),
                      "crops_with_under_10pct_of_box": round(float((cov < 0.1).float().mean()), 4),
                      "median_box_coverage": round(float(cov.median()), 4)}
    box_area = torch.tensor([(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) for i in has_box] or [1.0])

    # do the boxes find the organism? class-prototype top-1 on box vs same-size random crop
    labels_sorted = sorted(prototypes)
    proto_mat = torch.stack([prototypes[c] for c in labels_sorted]).to(device)
    col = {c: j for j, c in enumerate(labels_sorted)}
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
    check = {}
    for name, its in items.items():
        dl = DataLoader(Crops(args.data_root, rows, its, size, mean, std), batch_size=int(cfg.batch_size),
                        num_workers=int(cfg.num_workers))
        correct = total = 0
        for images, labels in dl:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                feats = model.forward_head(model.forward_features(images.to(device)), pre_logits=True)
            pred = (F.normalize(feats.float(), dim=-1) @ proto_mat.T).argmax(1).cpu()
            correct += sum(labels_sorted[int(p_)] == int(t) for p_, t in zip(pred, labels))
            total += len(labels)
        check[f"top1_{name}_crop"] = round(correct / max(total, 1), 4)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"boxes": {rows[i][0]: boxes[i] for i in has_box},
                "method": "DDT co-localization, pretrained backbone", "img_size": size,
                "margin": args.margin}, out / "boxes.pt")
    stats = {
        "images": len(rows), "classes": n_classes,
        "images_with_box": len(has_box),
        "images_without_box_usual_crops": len(rows) - len(has_box),
        "classes_where_prototype_sign_disagrees": sign_disagree,
        "box_area_fraction": {"mean": round(float(box_area.mean()), 4),
                              "median": round(float(box_area.median()), 4)},
        "random_crop_miss": miss,
        "box_check_prototype_top1": check,
        "box_finding_seconds": round(box_seconds, 1),
        "total_seconds": round(time.time() - started, 1),
        "device": str(device),
    }
    (out / "boxes_stats.json").write_text(json.dumps(stats, indent=2))
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
        fig.savefig(out / "boxes_overlay.png", dpi=110)
        print(f"[boxes] overlay -> {out / 'boxes_overlay.png'}")
    print(f"[boxes] {len(has_box)} of {len(rows)} images got a box -> {out / 'boxes.pt'}; "
          f"box finding took {box_seconds / 60:.1f} min (data preparation, report separately)")


if __name__ == "__main__":
    main()
