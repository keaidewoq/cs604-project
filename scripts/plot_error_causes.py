"""Slide figures: the error causes on the all-class holdout, as two charts.

All-class holdout, LoRA r24 no extras (lora_20261026_123804): 50,000 holdout photos (10,000 classes x 5),
11,111 wrong, 38,889 correct, 77.78% top-1. Counts come from
CS604_RUNS/allclass_holdout_analysis/causes/causes.json (scripts/error_causes.py):
flagged wrong = share_of_errors x 11,111, flagged correct = same_flag_among_correct_photos x 38,889.
A photo can have several flags.

  error_causes_share   of the 11,111 wrong photos, how many have each flag
  error_causes_ratio   of the photos with each flag, how many the model got wrong and right, against
                       all photos. Sibling is left out: it is defined by the wrong guess.

    python scripts/plot_error_causes.py --out_dir figures
writes <name>.png and <name>.svg for both.
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
UNSEEN = ("Unseen look", "photo unlike every training photo of its class", 0.5187, 0.1)
SIBLING = ("Sibling", "wrong guess is a close relative", 0.4985, None)
MESSY = ("Messy class", "class's training photos overlap another class", 0.7683, 0.5304)
TWO_LOOKS = ("Two looks", "photo shows the class's rarer look", 0.0484, 0.0225)
NO_FLAG = ("None of these", "no flag at all", 0.0392, None)
BAR_H, ROW_STEP, SEG_GAP = 0.56, 1.0, 0.35  # SEG_GAP in x units (%)


def setup(figsize, x_end, ys, top, bottom_pad):
    fig, ax = plt.subplots(figsize=figsize, facecolor=PAGE)
    ax.set_facecolor(PAGE)
    ax.set_xlim(0, x_end)
    bottom = ys[-1] - bottom_pad
    ax.set_ylim(bottom, top)
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    px_per_x, px_per_y = bbox.width / x_end, bbox.height / (top - bottom)
    return fig, ax, 6 / px_per_x, px_per_x / px_per_y


def seg(ax, x0, y, width, fill, edge, radius_x, aspect, z=2):
    ax.add_patch(FancyBboxPatch((x0, y - BAR_H / 2), width, BAR_H,
                                boxstyle=f"round,pad=0,rounding_size={min(radius_x, width / 2)}",
                                mutation_aspect=aspect, facecolor=fill, edgecolor=edge,
                                linewidth=1.0, zorder=z))


def finish(fig, ax, ys, rows, xlabel, footnote, out: Path):
    ax.set_yticks(ys, [f"{name}\n{meaning}" for name, meaning, *_ in rows])
    for t in ax.get_yticklabels():
        t.set_linespacing(1.3)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(xlabel, color=INK_2, labelpad=8, x=0, ha="left")
    ax.tick_params(axis="y", length=0, pad=12)
    ax.tick_params(axis="x", length=0, pad=6)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    fig.text(0.01, 0.025, footnote, color=INK_2, fontsize=10, linespacing=1.35)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=PAGE)
        print(f"[plot] wrote {out.with_suffix('.' + ext)}")
    plt.close(fig)


def share_figure(out: Path):
    rows = [MESSY, UNSEEN, SIBLING, TWO_LOOKS, NO_FLAG]
    ys = [-i * ROW_STEP for i in range(len(rows))]
    fig, ax, radius_x, aspect = setup((11, 5.4), 112, ys, 0.6, 0.6)
    for y, (_, _, ws, _) in zip(ys, rows):
        w = round(ws * WRONG)
        seg(ax, 0, y, 100, TRACK, TRACK, radius_x, aspect, z=1)
        seg(ax, 0, y, 100 * ws, *REMOVED, radius_x, aspect)
        if ws > 0.2:
            ax.text(1.5, y, f"{w:,} photos", va="center", ha="left", color=INK, fontsize=11.5, zorder=3)
            ax.text(100 * ws + 1.2, y, f"{100 * ws:.0f}%", va="center", ha="left", color=INK, fontweight="bold",
                    zorder=3)
        else:
            ax.text(100 * ws + 1.2, y, f"{100 * ws:.0f}%", va="center", ha="left", color=INK, fontweight="bold",
                    zorder=3)
            ax.text(100 * ws + 7, y, f"{w:,} photos", va="center", ha="left", color=INK, fontsize=11.5, zorder=3)
    finish(fig, ax, ys, rows, "Share of the 11,111 wrong photos",
           "A wrong photo can have several flags, so the bars add up to more than 100%.\n"
           "All-class holdout (10,000 classes, 5 photos each), LoRA r24. Flags use the model's own image features.",
           out)


def ratio_figure(out: Path):
    rows = [("All photos", "the baseline: every holdout photo", 1.0, 1.0), UNSEEN, TWO_LOOKS, MESSY]
    ys = [-i * ROW_STEP - (0.3 if i else 0) for i in range(len(rows))]  # extra space under the baseline
    top = 0.95
    fig, ax, radius_x, aspect = setup((12, 5.2), 130, ys, top, 0.6)
    for y, (_, _, ws, cs) in zip(ys, rows):
        w, c = round(ws * WRONG), round(cs * CORRECT)
        pw = 100 * w / (w + c)
        seg(ax, 0, y, pw, *REMOVED, radius_x, aspect)
        seg(ax, pw + SEG_GAP, y, 100 - pw - SEG_GAP, *NEUTRAL, radius_x, aspect)
        ax.text(1.5, y, f"{pw:.0f}% wrong", va="center", ha="left", color=INK, fontsize=12, fontweight="bold", zorder=5)
        ax.text(pw + 1.8, y, f"{100 - pw:.0f}% right", va="center", ha="left", color=INK_2, fontsize=12, zorder=5)
        ax.text(101.5, y + 0.12, f"{w + c:,} photos", va="center", ha="left", color=INK, fontsize=12.5,
                fontweight="bold")
        ax.text(101.5, y - 0.15, f"{w:,} wrong, {c:,} right", va="center", ha="left", color=INK_2, fontsize=10.5)

    base = 100 * WRONG / TOTAL
    ax.plot([base, base], [ys[-1] - 0.42, ys[0] - BAR_H / 2], color=INK, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.plot([0, 100], [(ys[0] + ys[1]) / 2] * 2, color=RULE, lw=0.8)
    for (fill, edge), label, x in ((REMOVED, "model wrong", 0), (NEUTRAL, "model right", 22)):
        ax.scatter([x + 0.8], [top - 0.2], s=150, marker="s", color=fill, edgecolors=edge, linewidths=1.0, clip_on=False)
        ax.text(x + 3.0, top - 0.2, label, va="center", ha="left", color=INK_2, fontsize=11)
    finish(fig, ax, ys, rows, "Of the photos in each row, the share the model got wrong and right",
           "Dashed line: the baseline, 22% wrong. A photo can have several flags, so the rows overlap. "
           "Sibling is not shown: it is defined by the wrong guess.\n"
           "All-class holdout (10,000 classes, 5 photos each), LoRA r24. Flags use the model's own image features.",
           out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="figures")
    args = p.parse_args()
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 13,
                         "text.color": INK, "xtick.color": INK_2, "ytick.color": INK})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    share_figure(out / "error_causes_share")
    ratio_figure(out / "error_causes_ratio")


if __name__ == "__main__":
    main()
