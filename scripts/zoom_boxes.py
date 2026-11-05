"""Where does Van's zoom look, and when does it change the answer? No training; GPU.

Runs the exact two-pass path of src.predict --zoom (src/van_zoom.py) on a labelled holdout, using a
checkpoint that never trained on it, and keeps each photo's box. The finals cannot be used here:
they trained on all 45 photos per class, and the val labels are hidden.

Writes to --out_dir:
  scan.csv            one row per photo: label, first-pass and after-zoom predictions, the box
  summary.json        top-1 before and after zoom, fixed / broken counts, box sizes
  candidates.json     photos worth showing, in three buckets (below), with their predictions and box
  candidates/<id>_view.jpg, <id>_zoom.jpg    what the model saw in each pass

Buckets, for choosing slide examples by eye; none says how often zoom helps (see summary.json):
  fixed_tight      zoom corrected the answer, tightest boxes first (small subjects)
  fixed_offcentre  zoom corrected the answer, box centre furthest from the image centre
  broken           zoom turned a right answer wrong, tightest boxes first

    python scripts/zoom_boxes.py --run_dir <run> --pilot_root /content/pilot_data/c10000_h5_s0 \\
        --out_dir <folder> --crop_pct 0.9
then scripts/plot_zoom_boxes.py draws the contact sheets and the slide figure (no GPU).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import van_zoom
from src.data import ImageListDataset
from src.model import build_model
from src.predict import load_checkpoint
from src.utils import apply_overrides, get_device, load_config, set_seed

BUCKETS = ("fixed_tight", "fixed_offcentre", "broken")
COLUMNS = ["index", "path", "label", "plain_pred", "plain_p", "final_pred", "final_p", "top", "left", "side"]


@torch.no_grad()
def scan(model, loader, device, use_bf16, mass, limit):
    """The same steps as van_zoom.zoom_view and predict_zoom.predict_batch, keeping the box."""
    size = int(model.patch_embed.img_size[0])
    patch = int(model.patch_embed.patch_size[0])
    grid = size // patch
    rows = []
    for (images, big), labels, paths in tqdm(loader, desc="zoom scan"):
        images, big = images.to(device, non_blocking=True), big.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            tokens = model.forward_features(images)
            logits = model.forward_head(tokens).float()
            boxes = van_zoom.select_boxes(van_zoom.cam_map(model, tokens, logits.argmax(1)), grid, mass,
                                          van_zoom.DEFAULT_MIN_SIDE)
            close_up = van_zoom.crop_zoom(big, boxes, patch * (big.shape[-1] // size), size)
            first = logits.softmax(1)
            final = (first + model(close_up).float().softmax(1)) / 2
        plain_p, plain_pred = first.max(1)
        final_p, final_pred = final.max(1)
        for i, path in enumerate(paths):
            top, left, side = boxes[i].tolist()
            rows.append({"index": len(rows), "path": path, "label": int(labels[i]),
                         "plain_pred": int(plain_pred[i]), "plain_p": round(float(plain_p[i]), 4),
                         "final_pred": int(final_pred[i]), "final_p": round(float(final_p[i]), 4),
                         "top": top, "left": left, "side": side})
        if limit and len(rows) >= limit:
            break
    return rows[:limit] if limit else rows, grid, patch


def summarise(rows, grid, mass, crop_pct):
    n = len(rows)
    plain = sum(r["plain_pred"] == r["label"] for r in rows)
    final = sum(r["final_pred"] == r["label"] for r in rows)
    fixed = sum(r["plain_pred"] != r["label"] and r["final_pred"] == r["label"] for r in rows)
    broken = sum(r["plain_pred"] == r["label"] and r["final_pred"] != r["label"] for r in rows)
    sides = sorted(r["side"] for r in rows)
    return {"photos": n, "crop_pct": crop_pct, "zoom_mass": mass, "grid": grid,
            "top1_plain": round(plain / n, 4), "top1_zoom": round(final / n, 4),
            "gain_points": round(100 * (final - plain) / n, 2),
            "fixed": fixed, "broken": broken,
            "answer_changed": sum(r["plain_pred"] != r["final_pred"] for r in rows),
            "box_side_patches": {"p10": sides[n // 10], "median": sides[n // 2], "p90": sides[9 * n // 10],
                                 "whole_view": sum(s == grid for s in sides)}}


def off_centre(row, grid):
    """Distance from the box centre to the view centre, in patches."""
    half = row["side"] / 2
    return ((row["top"] + half - grid / 2) ** 2 + (row["left"] + half - grid / 2) ** 2) ** 0.5


def choose(rows, grid, per_bucket):
    fixed = [r for r in rows if r["plain_pred"] != r["label"] and r["final_pred"] == r["label"]]
    broken = [r for r in rows if r["plain_pred"] == r["label"] and r["final_pred"] != r["label"]]
    tight = sorted(fixed, key=lambda r: (r["side"], r["index"]))[:per_bucket]
    used = {r["index"] for r in tight}
    away = sorted((r for r in fixed if r["index"] not in used), key=lambda r: (-off_centre(r, grid), r["index"]))
    return {"fixed_tight": tight, "fixed_offcentre": away[:per_bucket],
            "broken": sorted(broken, key=lambda r: (r["side"], r["index"]))[:per_bucket]}


def to_image(tensor, mean, std):
    pixels = (tensor.cpu() * std + mean).clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(pixels)


def normalisation(transform):
    for step in transform.transforms:
        if hasattr(step, "mean") and hasattr(step, "std"):
            return (torch.as_tensor(step.mean).view(3, 1, 1), torch.as_tensor(step.std).view(3, 1, 1))
    raise ValueError("no Normalize step in the eval transform")


def save_candidates(dataset, transform, buckets, patch, size, out_dir, crop_pct):
    """Re-read the chosen photos and save both views. The boxes are known, so no model is needed."""
    mean, std = normalisation(transform.small)
    image_dir = out_dir / "candidates"
    image_dir.mkdir(parents=True, exist_ok=True)
    chosen = {r["index"]: r for rows in buckets.values() for r in rows}
    records = {}
    for row, ((small, big), _, _) in zip(chosen.values(), Subset(dataset, list(chosen))):
        box = torch.tensor([[row["top"], row["left"], row["side"]]])
        close_up = van_zoom.crop_zoom(big[None], box, patch * (big.shape[-1] // size), size)[0]
        name = f"{row['index']:06d}"
        to_image(small, mean, std).save(image_dir / f"{name}_view.jpg", quality=92)
        to_image(close_up, mean, std).save(image_dir / f"{name}_zoom.jpg", quality=92)
        records[row["index"]] = {**row, "id": name, "box_px": [row["left"] * patch, row["top"] * patch,
                                                               row["side"] * patch], "view_px": size}
    payload = {"crop_pct": crop_pct, "buckets": {k: [records[r["index"]] for r in v] for k, v in buckets.items()}}
    (out_dir / "candidates.json").write_text(json.dumps(payload, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="folder with config.yaml and last.pt")
    p.add_argument("--pilot_root", required=True, help="split whose val.txt is the labelled holdout")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--crop_pct", type=float, default=0.9, help="0.9 for the 448 finals, 0.75 at 336")
    p.add_argument("--zoom_mass", type=float, default=van_zoom.DEFAULT_MASS)
    p.add_argument("--per_bucket", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="stop after this many photos (smoke test)")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    apply_overrides(cfg, data_root=str(args.pilot_root), batch_size=args.batch_size, num_workers=args.num_workers)
    device = get_device()
    set_seed(int(cfg.seed))
    model = build_model(cfg).to(device)
    load_checkpoint(model, run_dir / "last.pt", device)
    model.eval()

    transform = van_zoom.two_view_transform(cfg, args.crop_pct)
    dataset = ImageListDataset(args.pilot_root, "val", transform=transform)
    if not dataset.has_labels:
        raise SystemExit(f"{args.pilot_root}/val.txt has no labels: use a holdout split, not the official val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    use_bf16 = bool(cfg.bf16) and device.type == "cuda"
    rows, grid, patch = scan(model, loader, device, use_bf16, args.zoom_mass, args.limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "scan.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"run_dir": str(run_dir), **summarise(rows, grid, args.zoom_mass, args.crop_pct)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    save_candidates(dataset, transform, choose(rows, grid, args.per_bucket), patch, int(cfg.img_size), out_dir,
                    args.crop_pct)
    print(f"[zoom_boxes] wrote {out_dir}")


if __name__ == "__main__":
    main()
