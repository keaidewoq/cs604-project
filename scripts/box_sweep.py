"""Which way of finding the organism actually finds it? No training, about 10 minutes on an A100.

Runs several box-finding variants (src/boxes.py) on a subset of the pilot's classes and scores
each by how well the pretrained prototype head recognises its box crops, against crops of the
same size placed at random or in the centre (Van's zoom control). If a variant's boxes contain
the organism, its box crops should be recognised clearly more often than random ones; the first
pilot's boxes (ddt_final) managed only 0.591 vs 0.581.

One forward pass collects every layer the variants need, so all variants see the same photos.
--round picks the list of variants (ROUNDS below). Writes box_sweep_<round>.csv (one row per
variant), .json (with token-norm diagnostics) and _overlay.png (rows: photos, columns: variants).
Round 1's files (from before rounds existed) are box_sweep.csv / .json / _overlay.png.

The bar to clear is the centred crop of the same size: in round 1 it beat every box finder.

    python scripts/box_sweep.py --config configs/lora_r24_336px_qkv-proj-fc_no-extras_proto-cosine_2.4h.yaml \\
        --data_root /content/pilot_data/c1000_h5_s0 --out_dir <folder> [--round round2]
then rebuild the boxes with the winner: python scripts/find_boxes.py ... <its flags>
"""

from __future__ import annotations

import argparse
import csv
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

from scripts.find_boxes import load_backbone
from src.boxes import (FeatureTaps, WholeImages, box_from_map, class_scores, outlier_tokens, prototype_top1,
                       read_train_rows, variant_name)

ROUNDS = {
    # (score, layer, mask, smooth, top); layer = block index, None = final tokens;
    # top = keep the photo's top fraction of patches (None = all positive ones)
    "round1": [
        ("ddt", None, False, False, None),   # the first pilot's boxes (baseline)
        ("ddt", None, True, True, None),     # same, without outlier tokens, smoothed
        ("ddt", 9, False, True, None),       # earlier layer (block 10 of 12)
        ("ddt", 9, True, True, None),
        ("ddt", 6, False, True, None),       # middle layer (block 7 of 12)
        ("proto", None, False, True, None),  # true-label prototype map
        ("proto", None, True, True, None),
    ],
    # round 1: nothing beat random placement; the prototype map looked right but its
    # boxes were loose (median 71% of the photo). Tighter thresholds:
    "round2": [
        ("ddt", None, False, False, None),   # reference: the first pilot's boxes
        ("proto", None, False, True, None),  # reference: round 1's loose prototype boxes
        ("proto", None, False, True, 0.3),
        ("proto", None, False, True, 0.2),
        ("proto", None, False, False, 0.2),  # tight, without smoothing
        ("ddt", None, False, True, 0.2),     # the same tightening for DDT
    ],
}


def flags(v):
    out = [f"--score {v['score']}"]
    out += [f"--layer {v['layer']}"] if v["layer"] is not None else []
    out += ["--mask"] if v["mask"] else []
    out += ["--smooth"] if v["smooth"] else []
    out += [f"--top {v['top']}"] if v.get("top") else []
    return " ".join(out)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--classes", type=int, default=200, help="random subset of the data_root's classes")
    p.add_argument("--check_per_class", type=int, default=5)
    p.add_argument("--outlier_ratio", type=float, default=2.0)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--min_patches", type=int, default=4)
    p.add_argument("--max_area", type=float, default=0.8)
    p.add_argument("--overlay_photos", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--round", choices=sorted(ROUNDS), default="round2", help="which list of variants")
    p.add_argument("--out_name", default=None, help="file prefix (default box_sweep_<round>)")
    args = p.parse_args()

    started = time.time()
    cfg, device, model, mean, std = load_backbone(args.config)
    size = int(cfg.img_size)
    grid = size // int(model.patch_embed.patch_size[0])
    use_bf16 = bool(cfg.bf16) and device.type == "cuda"
    variants = [dict(score=s, layer=l, mask=m, smooth=sm, top=t) for s, l, m, sm, t in ROUNDS[args.round]]
    name = args.out_name or f"box_sweep_{args.round}"
    for v in variants:
        v["name"] = variant_name(v)
    taps = FeatureTaps(model, [v["layer"] for v in variants])

    rows = read_train_rows(args.data_root)
    classes = sorted({r[1] for r in rows})
    rng = random.Random(args.seed)
    keep = set(rng.sample(classes, min(args.classes, len(classes))))
    rows = [r for r in rows if r[1] in keep]
    print(f"[sweep] {len(variants)} variants on {len(rows)} photos of {len(keep)} classes, grid {grid}x{grid}")

    loader = DataLoader(WholeImages(args.data_root, rows, size, mean, std), batch_size=int(cfg.batch_size),
                        shuffle=False, num_workers=int(cfg.num_workers), pin_memory=device.type == "cuda")
    boxes = {v["name"]: [None] * len(rows) for v in variants}
    masked = {v["name"]: [] for v in variants}
    prototypes, maps_keep = {}, {}
    first_of_class = {}
    for i, (_, label) in enumerate(rows):
        first_of_class.setdefault(label, i)
    shown = sorted(first_of_class.values())[:: max(1, len(first_of_class) // args.overlay_photos)][:args.overlay_photos]
    norm_stats = {}  # layer -> [outlier fraction per class, max/median norm per class]
    buf = {"label": None, "idx": [], "feats": []}

    def flush():
        if not buf["idx"]:
            return
        feats = {k: torch.cat([f[k] for f in buf["feats"]]) for k in buf["feats"][0]}
        proto = F.normalize(feats["cls"].mean(0), dim=0)
        prototypes[buf["label"]] = proto
        for layer in [k for k in feats if k != "cls"]:
            norms = feats[layer].float().norm(dim=-1)
            s = norm_stats.setdefault("final" if layer == "final" else f"block{layer}", [[], []])
            s[0].append(float(outlier_tokens(feats[layer], args.outlier_ratio).float().mean()))
            s[1].append(float((norms.max(1).values / norms.median(1).values).mean()))
        for v in variants:
            scores, frac = class_scores(feats, v, grid, proto, args.outlier_ratio)
            masked[v["name"]].append(frac)
            for k, i in enumerate(buf["idx"]):
                boxes[v["name"]][i] = box_from_map(scores[k], grid, args.margin, args.min_patches, args.max_area,
                                                   v["top"])
                if i in shown:
                    maps_keep[(v["name"], i)] = scores[k].view(grid, grid).clone()
        buf.update(idx=[], feats=[])

    for images, labels, idx, _ in tqdm(loader, desc="features"):
        out = taps(images.to(device, non_blocking=True), use_bf16)
        for k in range(len(labels)):
            label, i = int(labels[k]), int(idx[k])
            if buf["label"] is not None and label != buf["label"]:
                flush()
            buf["label"] = label
            buf["idx"].append(i)
            buf["feats"].append({key: v[k:k + 1].half() if v.dim() == 3 else v[k:k + 1] for key, v in out.items()})
    flush()
    features_s = time.time() - started

    def top1(items):
        return prototype_top1(model, args.data_root, rows, items, prototypes, size, mean, std,
                              int(cfg.batch_size), int(cfg.num_workers), device, use_bf16)

    results = []
    whole_cache = {}
    for v in tqdm(variants, desc="checks"):
        bx = boxes[v["name"]]
        has = [i for i in range(len(rows)) if bx[i] is not None]
        per_class, items = {}, {"box": [], "random": [], "centre": []}
        crng = random.Random(args.seed)
        for i in has:
            label = rows[i][1]
            if per_class.get(label, 0) >= args.check_per_class:
                continue
            per_class[label] = per_class.get(label, 0) + 1
            b = bx[i]
            bw, bh = b[2] - b[0], b[3] - b[1]
            rx, ry = crng.uniform(0, 1 - bw), crng.uniform(0, 1 - bh)
            items["box"].append((i, b))
            items["random"].append((i, [rx, ry, rx + bw, ry + bh]))
            items["centre"].append((i, [0.5 - bw / 2, 0.5 - bh / 2, 0.5 + bw / 2, 0.5 + bh / 2]))
        acc = {name: top1(its) if its else float("nan") for name, its in items.items()}
        whole_key = tuple(i for i, _ in items["box"])
        if whole_key not in whole_cache:
            whole_cache[whole_key] = top1([(i, [0.0, 0.0, 1.0, 1.0]) for i in whole_key]) if whole_key else float("nan")
        areas = torch.tensor([(bx[i][2] - bx[i][0]) * (bx[i][3] - bx[i][1]) for i in has] or [float("nan")])
        results.append({
            "variant": v["name"], "flags": flags(v),
            "photos_with_box": round(len(has) / max(len(rows), 1), 4),
            "median_box_area": round(float(areas.median()), 4),
            "checked": len(items["box"]),
            "top1_box": round(acc["box"], 4), "top1_random_same_size": round(acc["random"], 4),
            "top1_centre_same_size": round(acc["centre"], 4), "top1_whole_photo": round(whole_cache[whole_key], 4),
            "lift_vs_random_points": round(100 * (acc["box"] - acc["random"]), 2),
            "lift_vs_centre_points": round(100 * (acc["box"] - acc["centre"]), 2),
            "outlier_tokens_masked": round(sum(masked[v["name"]]) / max(len(masked[v["name"]]), 1), 4) if v["mask"] else "",
        })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{name}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    diag = {layer: {"outlier_token_fraction": round(sum(s[0]) / len(s[0]), 4),
                    "mean_max_over_median_norm": round(sum(s[1]) / len(s[1]), 2)}
            for layer, s in norm_stats.items()}
    # the bar to clear is the centred crop of the same size (photographers centre the subject)
    best = max(results, key=lambda r: (r["lift_vs_centre_points"] if r["photos_with_box"] >= 0.5 else -1e9))
    summary = {"round": args.round, "photos": len(rows), "classes": len(keep), "token_norms": diag, "results": results,
               "best_vs_centre_with_boxes_on_half_the_photos": best["variant"],
               "seconds": {"features": round(features_s, 1), "total": round(time.time() - started, 1)}}
    (out / f"{name}.json").write_text(json.dumps(summary, indent=2))

    print("\n[sweep] token norms (outlier = norm > %.1f x the photo's median):" % args.outlier_ratio)
    for layer, d in diag.items():
        print(f"  {layer:8s} outlier tokens {d['outlier_token_fraction']:.2%}, "
              f"max/median norm {d['mean_max_over_median_norm']:.2f}")
    print("\n[sweep] variant                      box%   area  top1 box  random  centre  whole  vs rand  vs centre")
    for r in sorted(results, key=lambda r: -r["lift_vs_centre_points"]):
        print(f"  {r['variant']:28s} {r['photos_with_box']:5.0%}  {r['median_box_area']:5.2f}   "
              f"{r['top1_box']:.3f}   {r['top1_random_same_size']:.3f}   {r['top1_centre_same_size']:.3f}  "
              f"{r['top1_whole_photo']:.3f}   {r['lift_vs_random_points']:+6.1f}   {r['lift_vs_centre_points']:+6.1f}")
    print(f"\n[sweep] best vs the centred crop, with boxes on at least half the photos: {best['variant']} "
          f"({best['lift_vs_centre_points']:+.1f} points vs centre, {best['lift_vs_random_points']:+.1f} vs random); "
          f"rebuild with: find_boxes.py ... {best['flags']}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(len(shown), len(variants), figsize=(2.6 * len(variants), 2.6 * len(shown)),
                             squeeze=False)
    for r, i in enumerate(shown):
        image = Image.open(Path(args.data_root) / rows[i][0]).convert("RGB").resize((size, size))
        for c, v in enumerate(variants):
            ax = axes[r, c]
            ax.axis("off")
            ax.imshow(image)
            heat = maps_keep[(v["name"], i)].clamp_min(0)
            ax.imshow(F.interpolate(heat[None, None], size=(size, size), mode="bilinear")[0, 0],
                      alpha=0.45, cmap="jet", extent=(0, size, size, 0))
            b = boxes[v["name"]][i]
            if b:
                ax.add_patch(Rectangle((b[0] * size, b[1] * size), (b[2] - b[0]) * size, (b[3] - b[1]) * size,
                                       fill=False, edgecolor="lime", linewidth=2))
            if r == 0:
                ax.set_title(v["name"], fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"{name}_overlay.png", dpi=100)
    print(f"[sweep] wrote {name}.csv / .json / _overlay.png in {out} "
          f"({(time.time() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
