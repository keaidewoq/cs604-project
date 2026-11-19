"""Slide figure: real zoom boxes, from the folder scripts/zoom_boxes.py wrote. No GPU, no model.

First draw the contact sheets (sheet_<bucket>.png, one per bucket, each photo labelled with its id),
look at them, then draw the figure from the ids you pick:

    python scripts/plot_zoom_boxes.py --in_dir <folder> --sheets
    python scripts/plot_zoom_boxes.py --in_dir <folder> --picks 001234 004567 009876 \\
        --model "Full FT at 448 px, all-class holdout" --out figures/zoom_boxes
writes <out>.png and <out>.svg. Species names come from outputs/inat21_class_names.csv, class IDs without it.
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw

PAGE, INK, INK_2 = "#fbfaf7", "#3d3d3a", "#5f5e5a"
RIGHT_INK, WRONG_INK, BOX = "#0f6e56", "#b5441f", "#e5322d"
SHEET_COLUMNS = 4


def species_names(path):
    if not Path(path).is_file():
        return {}
    with open(path, encoding="utf8") as f:
        rows = {int(r["class"]): r for r in csv.DictReader(f)}
    return {k: (r["common_name"] or r["name"])[0].upper() + (r["common_name"] or r["name"])[1:]
            for k, r in rows.items()}


def load(in_dir, record):
    folder = Path(in_dir) / "candidates"
    return (Image.open(folder / f"{record['id']}_view.jpg").convert("RGB"),
            Image.open(folder / f"{record['id']}_zoom.jpg").convert("RGB"))


def sheets(in_dir, candidates, names):
    def name(class_id):
        return names.get(class_id, str(class_id))

    for bucket, records in candidates["buckets"].items():
        if not records:
            print(f"{bucket}: no candidates")
            continue
        rows = (len(records) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
        fig, axes = plt.subplots(rows, SHEET_COLUMNS, figsize=(4.2 * SHEET_COLUMNS, 3.1 * rows), facecolor="white",
                                 squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, r in zip(axes.flat, records):
            view, zoom = load(in_dir, r)
            pair = Image.new("RGB", (view.width + zoom.width + 8, view.height), "white")
            pair.paste(view, (0, 0))
            pair.paste(zoom, (view.width + 8, 0))
            left, top, side = r["box_px"]
            ImageDraw.Draw(pair).rectangle([left, top, left + side - 1, top + side - 1], outline=BOX, width=4)
            ax.imshow(pair)
            ax.set_title(f"{r['id']}  box {r['side']} patches\ntrue {name(r['label'])}\n"
                         f"1st {name(r['plain_pred'])} {r['plain_p']:.0%} > zoom {name(r['final_pred'])} "
                         f"{r['final_p']:.0%}", fontsize=8, loc="left")
        fig.tight_layout()
        out = Path(in_dir) / f"sheet_{bucket}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"wrote {out}")


def figure(in_dir, candidates, names, picks, model_label, out):
    by_id = {r["id"]: r for records in candidates["buckets"].values() for r in records}
    missing = [i for i in picks if i not in by_id]
    if missing:
        raise SystemExit(f"not in candidates.json: {missing}")
    if not picks:
        raise SystemExit("give at least one id with --picks")

    def name(class_id):
        return names.get(class_id, f"class {class_id}")

    def verdict(fig, x, y, text, right, width):
        lines = textwrap.wrap(text, width)
        for i, line in enumerate(lines):
            fig.text(x / fig_w, (y - i * 0.2) / fig_h, line, va="baseline", ha="left", fontsize=10.5,
                     fontweight="bold", color=RIGHT_INK if right else WRONG_INK)
        return y - len(lines) * 0.2 - 0.05

    cell_w, gap, image = 3.85, 0.1, 1.82
    fig_w, fig_h = 0.25 + len(picks) * (cell_w + 0.1), 3.8
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "text.color": INK})
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=PAGE)
    image_y = fig_h - 0.42 - image
    for n, i in enumerate(picks):
        r = by_id[i]
        x = 0.2 + n * (cell_w + 0.1)
        view, zoom = load(in_dir, r)
        for column, (photo, caption) in enumerate(((view, "First pass (red = chosen box)"), (zoom, "Second pass"))):
            ax_x = x + column * (image + gap)
            ax = fig.add_axes([ax_x / fig_w, image_y / fig_h, image / fig_w, image / fig_h])
            ax.imshow(photo, interpolation="lanczos")
            ax.axis("off")
            if column == 0:
                left, top, side = r["box_px"]
                ax.add_patch(Rectangle((left, top), side, side, fill=False, edgecolor=BOX, linewidth=2))
            fig.text(ax_x / fig_w, (image_y + image + 0.08) / fig_h, caption, va="baseline", ha="left",
                     fontsize=10, color=INK_2)
        y = image_y - 0.25
        fig.text(x / fig_w, y / fig_h, f"True: {name(r['label'])}", va="baseline", ha="left", fontsize=10.5,
                 color=INK)
        y -= 0.23
        y = verdict(fig, x, y, f"First pass: {name(r['plain_pred'])} ({r['plain_p']:.0%})",
                    r["plain_pred"] == r["label"], 44)
        verdict(fig, x, y, f"With zoom: {name(r['final_pred'])} ({r['final_p']:.0%})",
                r["final_pred"] == r["label"], 44)
    fig.text(0.2 / fig_w, 0.07 / fig_h,
             f"Selected examples where zoom changed the answer, not how often it does. {model_label}. "
             "% = model confidence; with zoom, the average of both passes.",
             va="baseline", ha="left", fontsize=8.5, color=INK_2)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=PAGE)
    fig.savefig(out.with_suffix(".svg"), facecolor=PAGE)
    print(f"wrote {out}.png and .svg")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", required=True, help="the --out_dir of scripts/zoom_boxes.py")
    p.add_argument("--names", default="outputs/inat21_class_names.csv")
    p.add_argument("--sheets", action="store_true", help="draw the contact sheets")
    p.add_argument("--picks", nargs="*", default=None, help="ids for the figure, shown left to right")
    p.add_argument("--model", default="Full FT at 448 px, all-class holdout", help="model named in the footnote")
    p.add_argument("--out", default="figures/zoom_boxes", help="output path without extension")
    args = p.parse_args()
    if not args.sheets and args.picks is None:
        p.error("give --sheets, --picks or both")

    candidates = json.loads((Path(args.in_dir) / "candidates.json").read_text())
    names = species_names(args.names)
    if args.sheets:
        sheets(args.in_dir, candidates, names)
    if args.picks is not None:
        figure(args.in_dir, candidates, names, args.picks, args.model, args.out)


if __name__ == "__main__":
    main()
