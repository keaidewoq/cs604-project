"""Slide figures for kNN re-ranking: how it works, and which re-rankers were compared.

  knn_how_it_works  the method step by step. A schematic: classes are letters and bar lengths are
                    illustrative. The settings are the real ones (K = 5, m = 3, beta = 20).
  knn_variants      the hand-made re-rankers of scripts/rerank_bakeoff.py, each with a sketch of what
                    it compares the photo against. Gains are measured (progress_log.md, 2026-10-27:
                    LoRA r24 no extras, all-class holdout, beta chosen on the other half of the
                    classes); the sketches are not data.

    python scripts/plot_knn_diagrams.py --out_dir figures
writes <name>.png and <name>.svg for each.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_full_ft_ablation import INK, INK_2, NEUTRAL, PAGE, RECIPE, RULE

QUERY = "#d85a30"
OTHER_CLASS = "#b4b2a9"

# The schematic's two signals, per class. Illustrative lengths, not measurements.
MODEL = {"A": 0.90, "B": 0.70, "C": 0.35, "D": 0.25, "E": 0.15}
MATCH = {"A": 0.40, "B": 0.85, "C": 0.45, "D": 0.20, "E": 0.35}
TILES = {"A": (0.55, 0.40, 0.25), "B": (0.95, 0.85, 0.75), "C": (0.60, 0.45, 0.30),
         "D": (0.30, 0.20, 0.10), "E": (0.50, 0.35, 0.20)}

# (group, name, what it compares with, gain plain view, gain on zoom, sketch)
VARIANTS = [
    ("Compares with individual training photos", "kNN mean of 3 (chosen)", "Mean cosine to the class's 3 closest photos",
     1.32, 1.23, "mean3"),
    (None, "kNN best match", "Cosine to the class's single closest photo", 1.18, 1.02, "max"),
    (None, "Weighted vote", "The photo's 50 nearest photos, any class, vote", 1.14, 1.03, "vote"),
    ("Compares with a summary of the class", "3 k-means centres", "Best cosine to 3 centres of the class's photos",
     0.70, 0.69, "centres"),
    (None, "Class mean (NCM)", "Cosine to the class's average feature", 0.55, 0.49, "ncm"),
    (None, "kNN mean of all", "Mean cosine to all of the class's photos", 0.13, 0.13, "meanall"),
]
# Sketch layout: a class with a large group of photos and a small one, and a photo near the small one.
CLASS_PHOTOS = [(0.16, 0.30), (0.25, 0.52), (0.32, 0.22), (0.38, 0.42), (0.46, 0.27), (0.21, 0.72), (0.44, 0.60),
                (0.80, 0.78), (0.91, 0.62), (0.70, 0.88)]
OTHER_PHOTOS = [(0.62, 0.40), (0.97, 0.90), (0.60, 0.14)]
PHOTO = (0.76, 0.58)


def setup(width, height):
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 12,
                         "text.color": INK})
    fig = plt.figure(figsize=(width, height), facecolor=PAGE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def bar(ax, x, y, length, height, style, radius=0.04):
    if length <= 0:
        return
    ax.add_patch(FancyBboxPatch((x, y), length, height, boxstyle=f"round,pad=0,rounding_size={min(radius, length / 2)}",
                                facecolor=style[0], edgecolor=style[1], linewidth=1.0, zorder=3))


def box(ax, x, y, w, h, lines, style=NEUTRAL, size=12):
    bar(ax, x, y, w, h, style, radius=0.08)
    ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center", fontsize=size, linespacing=1.3, zorder=4)


def arrow(ax, start, end, curve=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color=INK_2, linewidth=1.3,
                                 connectionstyle=f"arc3,rad={curve}", shrinkA=0, shrinkB=0, zorder=2))


def shade(strength):
    """Sequential one-hue ramp for how close a training photo is: pale to dark green."""
    pale, dark = (0.882, 0.961, 0.933), (0.059, 0.431, 0.337)
    return tuple(a + (b - a) * strength for a, b in zip(pale, dark))


def how_it_works(out):
    width, height = 12.0, 5.25
    fig, ax = setup(width, height)
    classes = list(MODEL)
    rows = {c: 3.62 - i * 0.5 for i, c in enumerate(classes)}
    head = 4.92

    # inputs
    box(ax, 0.2, 3.3, 1.55, 0.72, ["Photo"])
    box(ax, 0.2, 2.05, 1.55, 0.72, ["Trained", "model"])
    arrow(ax, (0.975, 3.3), (0.975, 2.77))
    box(ax, 5.05, 0.48, 2.85, 0.66, ["Bank: features of every training", "photo, built once by the same model"], size=10.5)
    arrow(ax, (6.48, 1.14), (6.48, 1.57))
    ax.plot([0.975, 0.975], [2.05, 0.81], color=INK_2, linewidth=1.3, zorder=2)
    arrow(ax, (0.975, 0.81), (5.05, 0.81))
    ax.text(1.1, 0.95, "the photo's feature", fontsize=10.5, color=INK_2, va="center")

    def title(x, step, text, note):
        ax.text(x, head, f"{step}. {text}", fontsize=13.5, fontweight="bold", va="center")
        ax.text(x, head - 0.34, note, fontsize=10.5, color=INK_2, va="center")

    # 1: the model's short list
    x1 = 2.3
    title(x1, 1, "Model's top 5 classes", "by log p (with zoom: averaged p)")
    for c in classes:
        ax.text(x1, rows[c] + 0.16, c, fontsize=12, va="center", fontweight="bold")
        bar(ax, x1 + 0.3, rows[c], MODEL[c] * 2.0, 0.32, NEUTRAL)
    arrow(ax, (1.75, 2.41), (x1 - 0.1, 2.55))
    arrow(ax, (4.45, 2.55), (4.95, 2.55))

    # 2: the closest training photos of each candidate
    x2 = 5.05
    title(x2, 2, "Closest training photos", "3 best cosines per class, then average")
    for c in classes:
        ax.text(x2, rows[c] + 0.16, c, fontsize=12, va="center", fontweight="bold")
        for i, strength in enumerate(TILES[c]):
            ax.add_patch(FancyBboxPatch((x2 + 0.3 + i * 0.38, rows[c]), 0.32, 0.32, boxstyle="round,pad=0,rounding_size=0.04",
                                        facecolor=shade(strength), edgecolor=RECIPE[1], linewidth=0.8, zorder=3))
        bar(ax, x2 + 1.55, rows[c], MATCH[c] * 1.3, 0.32, RECIPE)
    ax.text(x2 + 0.3, rows["E"] - 0.2, "darker = closer", fontsize=9.5, color=INK_2, va="center")
    ax.text(x2 + 1.55, rows["E"] - 0.2, "mean of the 3", fontsize=9.5, color=INK_2, va="center")

    # 3: new score, re-sorted
    x3 = 8.45
    title(x3, 3, "Re-score and re-sort", "score = log p + 20 × mean cosine")
    order = sorted(classes, key=lambda c: -(MODEL[c] + MATCH[c]))
    scale = 1.35
    for i, c in enumerate(order):
        y = 3.62 - i * 0.5
        ax.text(x3, y + 0.16, c, fontsize=12, va="center", fontweight="bold")
        bar(ax, x3 + 0.3, y, MODEL[c] * scale, 0.32, NEUTRAL)
        bar(ax, x3 + 0.3 + MODEL[c] * scale + 0.03, y, MATCH[c] * scale, 0.32, RECIPE)
    arrow(ax, (x2 + 1.55 + 0.85 * 1.3 + 0.12, 2.55), (x3 - 0.08, 2.55))
    ax.text(x3 + 0.3 + (MODEL["B"] + MATCH["B"]) * scale + 0.1, 3.78, "Prediction", fontsize=11, fontweight="bold",
            va="center", ha="left")
    # key for the stacked bars
    key_y = 1.0
    bar(ax, x3 + 0.3, key_y, 0.22, 0.16, NEUTRAL, radius=0.03)
    ax.text(x3 + 0.58, key_y + 0.08, "model", fontsize=10, color=INK_2, va="center")
    bar(ax, x3 + 1.2, key_y, 0.22, 0.16, RECIPE, radius=0.03)
    ax.text(x3 + 1.48, key_y + 0.08, "photo match", fontsize=10, color=INK_2, va="center")

    ax.text(0.2, 0.18, "Schematic: classes are letters and bar lengths are illustrative. Real settings: top 5, 3 best cosines, "
            "weight 20. No training and no trainable parameters.", fontsize=10, color=INK_2, va="center")
    save(fig, out)


def sketch(ax, x, y, w, h, kind):
    def at(point):
        return x + point[0] * w, y + point[1] * h

    def link(point, weight=1.1):
        (ax0, ay0), (ax1, ay1) = at(PHOTO), at(point)
        ax.plot([ax0, ax1], [ay0, ay1], color=RECIPE[1], linewidth=weight, zorder=2, solid_capstyle="round")

    def dist(point):
        return (point[0] - PHOTO[0]) ** 2 + (point[1] - PHOTO[1]) ** 2

    nearest = sorted(CLASS_PHOTOS, key=dist)
    used = []
    summaries = []
    if kind == "mean3":
        used = nearest[:3]
    elif kind == "max":
        used = nearest[:1]
    elif kind == "vote":
        used = sorted(CLASS_PHOTOS + OTHER_PHOTOS, key=dist)[:5]
    elif kind == "meanall":
        used = CLASS_PHOTOS
    elif kind == "ncm":
        summaries = [(sum(p[0] for p in CLASS_PHOTOS) / len(CLASS_PHOTOS), sum(p[1] for p in CLASS_PHOTOS) / len(CLASS_PHOTOS))]
    elif kind == "centres":
        summaries = [(0.24, 0.40), (0.43, 0.42), (0.81, 0.76)]
    lines = [(p, 0.6 if kind == "meanall" else 1.1) for p in used]
    if kind == "ncm":
        lines = [(summaries[0], 1.1)]
    elif kind == "centres":
        lines = [(summaries[2], 1.1)]
    for point, weight in lines:
        link(point, weight)

    for point in CLASS_PHOTOS:
        px, py = at(point)
        chosen = point in used
        ax.plot(px, py, "o", markersize=6 if chosen else 5, markerfacecolor=RECIPE[1] if chosen else RULE,
                markeredgecolor=PAGE, markeredgewidth=0.8, zorder=3)
    if kind == "vote":
        for point in OTHER_PHOTOS:
            px, py = at(point)
            ax.plot(px, py, "^", markersize=6, markerfacecolor=RECIPE[1] if point in used else OTHER_CLASS,
                    markeredgecolor=PAGE, markeredgewidth=0.8, zorder=3)
    for point in summaries:
        px, py = at(point)
        ax.plot(px, py, "D", markersize=7, markerfacecolor=RECIPE[1], markeredgecolor=PAGE, markeredgewidth=0.8, zorder=4)
    px, py = at(PHOTO)
    ax.plot(px, py, "*", markersize=13, markerfacecolor=QUERY, markeredgecolor=PAGE, markeredgewidth=0.8, zorder=5)


def variants(out):
    width, height = 12.0, 7.3
    fig, ax = setup(width, height)
    bars_x, bars_w, axis_max = 6.15, 3.6, 1.5
    row_h, y = 0.74, 6.5
    ax.text(0.25, 6.97, "What it compares the photo with", fontsize=11, color=INK_2, va="center")
    ax.text(bars_x, 6.97, "Gain over the model alone (points)", fontsize=11, color=INK_2, va="center")
    ax.text(10.65, 6.97, "On top of zoom", fontsize=11, color=INK_2, va="center")

    top = None
    for group, name, description, gain, zoom_gain, kind in VARIANTS:
        if group:
            y -= 0.2
            ax.text(0.25, y, group, fontsize=12, fontweight="bold", va="center")
            ax.plot([0.25, 11.75], [y - 0.18, y - 0.18], color=RULE, linewidth=0.6)
            y -= 0.28
        y -= row_h
        top = top or y + row_h
        chosen = kind == "mean3"
        sketch(ax, 0.25, y + 0.06, 1.15, row_h - 0.12, kind)
        ax.text(1.6, y + row_h * 0.64, name, fontsize=12.5, fontweight="bold" if chosen else "normal", va="center")
        ax.text(1.6, y + row_h * 0.28, description, fontsize=10, color=INK_2, va="center")
        bar(ax, bars_x, y + 0.17, gain / axis_max * bars_w, row_h - 0.34, RECIPE if chosen else NEUTRAL)
        ax.text(bars_x + gain / axis_max * bars_w + 0.08, y + row_h / 2, f"+{gain:.2f}", fontsize=12, fontweight="bold",
                va="center")
        ax.text(10.65, y + row_h / 2, f"+{zoom_gain:.2f}", fontsize=12, color=INK, va="center")

    # sketch key
    key_y = y - 0.2
    items = [("*", QUERY, 11, "the photo"), ("o", RULE, 5, "training photos of the class"), ("^", OTHER_CLASS, 6, "another class"),
             ("D", RECIPE[1], 6, "class summary"), ("o", RECIPE[1], 6, "what it compares with")]
    x = 0.3
    for marker, colour, size, label in items:
        ax.plot(x, key_y, marker, markersize=size, markerfacecolor=colour, markeredgecolor=PAGE, markeredgewidth=0.6)
        ax.text(x + 0.12, key_y, label, fontsize=9.5, color=INK_2, va="center")
        x += 0.36 + len(label) * 0.066
    ax.text(0.25, key_y - 0.3, "Sketches are not data. Gains: LoRA r24 no extras, 50,000 all-class holdout photos, top 5 candidates, "
            "weight chosen on the other half of the classes.", fontsize=9.5, color=INK_2, va="center")
    ax.text(0.25, key_y - 0.52, "kNN settings tried: top 2 / 3 / 5, 1 or 3 matches, weight 0 to 40; both halves chose top 5, 3, 20. "
            "Learned linear re-ranker: +0.20 over kNN mean of 3 after zoom, not kept.", fontsize=9.5, color=INK_2, va="center")
    save(fig, out)


def save(fig, out):
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=PAGE)
    fig.savefig(out.with_suffix(".svg"), facecolor=PAGE)
    plt.close(fig)
    print(f"wrote {out}.png and .svg")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="figures")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    how_it_works(out_dir / "knn_how_it_works")
    variants(out_dir / "knn_variants")


if __name__ == "__main__":
    main()
