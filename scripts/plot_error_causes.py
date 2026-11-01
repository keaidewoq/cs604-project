"""Slide figure: how many of the 50,000 holdout photos have each error-cause flag, and how many of those are wrong.

All-class holdout, LoRA r24 no extras (lora_20261026_123804): 50,000 holdout photos (10,000 classes x 5),
11,111 wrong, 38,889 correct, 77.78% top-1. One bar per flag: its length is the share of all 50,000 photos
with the flag, split into the photos the model got wrong and right. The top bar is all photos, the baseline.
Counts come from CS604_RUNS/allclass_holdout_analysis/causes/causes.json (scripts/error_causes.py):
flagged wrong = share_of_errors x 11,111, flagged correct = same_flag_among_correct_photos x 38,889.
Sibling is defined by the wrong guess, so all its photos are wrong. A photo can have several flags.

    python scripts/plot_error_causes.py --out figures/error_causes
writes <out>.png and <out>.svg.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from plot_full_ft_ablation import INK, INK_2, NEUTRAL, PAGE, REMOVED, RULE

WRONG, CORRECT = 11_111, 38_889
TOTAL = WRONG + CORRECT
TRACK = "#efede6"  # the empty 0-100% track behind each bar

# (cause, what the flag means, share of wrong photos flagged, share of correct photos flagged or None)
CAUSES = [
    ("All photos", "the baseline: every holdout photo", 1.0, 1.0),
    ("Unseen look", "photo unlike every training photo of its class", 0.5187, 0.1),
    ("Sibling", "wrong guess is a close relative", 0.4985, None),
    ("Messy class", "class's training photos overlap another class", 0.7683, 0.5304),
    ("Two looks", "photo shows the class's rarer look", 0.0484, 0.0225),
]
BAR_H, ROW_STEP, SEG_GAP = 0.56, 1.0, 0.35  # SEG_GAP in x units (% of photos)


def seg(ax, x0, y, width, fill, edge, radius_x, aspect, z=2):
    ax.add_patch(FancyBboxPatch((x0, y - BAR_H / 2), width, BAR_H,
                                boxstyle=f"round,pad=0,rounding_size={min(radius_x, width / 2)}",
                                mutation_aspect=aspect, facecolor=fill, edgecolor=edge,
                                linewidth=1.0, zorder=z))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="figures/error_causes", help="output path without extension")
    args = p.parse_args()

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 13,
                         "text.color": INK, "xtick.color": INK_2, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(12, 6.4), facecolor=PAGE)
    ax.set_facecolor(PAGE)
    x_end, top = 126, 1.0
    ax.set_xlim(0, x_end)
    ys = [-i * ROW_STEP - (0.35 if i else 0) for i in range(len(CAUSES))]  # extra space under the baseline
    bottom = ys[-1] - 0.6
    ax.set_ylim(bottom, top)

    fig.canvas.draw()
    bbox = ax.get_window_extent()
    px_per_x, px_per_y = bbox.width / x_end, bbox.height / (top - bottom)
    radius_x, aspect = 6 / px_per_x, px_per_x / px_per_y

    for y, (name, _, ws, cs) in zip(ys, CAUSES):
        w = round(ws * WRONG)
        c = 0 if cs is None else round(cs * CORRECT)
        pw, pc = 100 * w / TOTAL, 100 * c / TOTAL
        seg(ax, 0, y, 100, TRACK, TRACK, radius_x, aspect, z=1)
        seg(ax, 0, y, pw, *REMOVED, radius_x, aspect)
        if c:
            seg(ax, pw + SEG_GAP, y, pc - SEG_GAP, *NEUTRAL, radius_x, aspect)
        # counts inside the segments when they fit, else just after the bar
        if pw > 15:
            ax.text(1.2, y, f"{w:,} wrong", va="center", ha="left", color=INK, fontsize=11, zorder=3)
            ax.text(pw + 1.5, y, f"{c:,} right", va="center", ha="left", color=INK_2, fontsize=11, zorder=3)
        else:
            text = f"{w:,} wrong" + (f", {c:,} right" if c else "")
            ax.text(pw + pc + 1.2, y, text, va="center", ha="left", color=INK_2, fontsize=11, zorder=3)
        # how many photos have the flag, and how many of them are wrong
        ax.text(101.5, y + 0.11, f"{100 * (w + c) / TOTAL:.0f}% of photos", va="center", ha="left",
                color=INK, fontsize=12.5, fontweight="bold")
        rate = "all wrong (by definition)" if cs is None else f"{100 * w / (w + c):.0f}% of them wrong"
        ax.text(101.5, y - 0.15, rate, va="center", ha="left", color=INK_2, fontsize=11)

    ax.plot([0, 100], [(ys[0] + ys[1]) / 2] * 2, color=RULE, lw=0.8)
    ax.text(101.5, top - 0.22, "vs baseline 22% wrong", va="center", ha="left", color=INK_2,
            fontsize=10.5, style="italic")

    for (fill, edge), label, x in ((REMOVED, "model wrong", 0), (NEUTRAL, "model right", 22)):
        ax.scatter([x + 0.8], [top - 0.22], s=150, marker="s", color=fill, edgecolors=edge, linewidths=1.0,
                   clip_on=False)
        ax.text(x + 3.0, top - 0.22, label, va="center", ha="left", color=INK_2, fontsize=11)

    ax.set_yticks(ys, [f"{name}\n{meaning}" for name, meaning, _, _ in CAUSES])
    labels = ax.get_yticklabels()
    for t in labels:
        t.set_linespacing(1.3)
    labels[0].set_fontweight("bold")
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of all 50,000 holdout photos", color=INK_2, labelpad=8)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.tick_params(axis="x", length=0, pad=6)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    fig.text(0.01, 0.025, "All-class holdout (10,000 classes, 5 photos each), LoRA r24, 77.8% right. A photo can "
             "have several flags, so the bars overlap and do not add up.\nFlags are computed from the model's "
             "own image features.", color=INK_2, fontsize=10, linespacing=1.35)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=PAGE)
        print(f"[plot] wrote {out.with_suffix('.' + ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
