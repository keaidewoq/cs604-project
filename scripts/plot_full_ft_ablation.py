"""Slide figure: Van's full-FT recipe ablation (Van holdout, 3 epochs, one change at a time).

One control; "- remove layer decay" is tested against the control, the "+" rows build on the
row above. Numbers from report/presentation_storyboard.md (slide 6) / progress_log.md (2026-10-06).

    python scripts/plot_full_ft_ablation.py --out figures/full_ft_ablation
writes <out>.png and <out>.svg.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PAGE, INK, INK_2, RULE = "#fbfaf7", "#3d3d3a", "#5f5e5a", "#b4b2a9"
# (fill, outline) pairs, light pastel fill with a darker edge
NEUTRAL = ("#f1efe8", "#888780")
REMOVED = ("#faece7", "#d85a30")
RECIPE = ("#e1f5ee", "#1d9e75")

# (label, top-1, change vs the reference row, style)
ROWS = [
    ("Control (LP-FT start, layer decay)", 69.83, None, NEUTRAL),
    ("− remove layer decay", 63.12, 63.12 - 69.83, REMOVED),
    ("+ head lr 10× backbone", 71.37, 71.37 - 69.83, RECIPE),
    ("+ prototype head start", 72.98, 72.98 - 71.37, RECIPE),
    ("+ train 6 epochs (not 3)", 73.97, 73.97 - 72.98, NEUTRAL),
    ("+ train 12 epochs", 74.49, 74.49 - 73.97, NEUTRAL),
]
AXIS_START, AXIS_END = 60, 77
GAP_AFTER = 1          # dashed rule after this row: removal above, build-up below
BAR_H, ROW_STEP, GAP = 0.62, 1.0, 0.45


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="figures/full_ft_ablation", help="output path without extension")
    args = p.parse_args()

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 13,
                         "text.color": INK, "xtick.color": INK_2, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor=PAGE)
    ax.set_facecolor(PAGE)
    ax.set_xlim(AXIS_START, AXIS_END)

    ys, y = [], 0.0
    for i in range(len(ROWS)):
        ys.append(-y)
        y += ROW_STEP + (GAP if i == GAP_AFTER else 0)
    ax.set_ylim(ys[-1] - 0.6, 0.6)

    # round the bar ends by the same number of pixels in x and y
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    px_per_x = bbox.width / (AXIS_END - AXIS_START)
    px_per_y = bbox.height / (0.6 - (ys[-1] - 0.6))
    radius_x = 7 / px_per_x
    for y, (label, acc, delta, (fill, edge)) in zip(ys, ROWS):
        ax.add_patch(FancyBboxPatch((AXIS_START, y - BAR_H / 2), acc - AXIS_START, BAR_H,
                                    boxstyle=f"round,pad=0,rounding_size={radius_x}",
                                    mutation_aspect=px_per_x / px_per_y,
                                    facecolor=fill, edgecolor=edge, linewidth=1.0, zorder=2))
        text = f"{acc:.1f}" if delta is None else f"{acc:.1f} ({delta:+.1f})".replace("-", "−")
        ax.text(acc + 0.3, y, text, va="center", ha="left", color=INK)

    rule_y = (ys[GAP_AFTER] + ys[GAP_AFTER + 1]) / 2
    ax.plot([-0.28, 0.95], [rule_y, rule_y], transform=ax.get_yaxis_transform(), clip_on=False,
            color=RULE, lw=0.8, ls=(0, (4, 4)))

    ax.set_yticks(ys, [r[0] for r in ROWS])
    ax.set_xticks(range(AXIS_START, AXIS_END, 4))
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", length=0, pad=8)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["left"].set_linewidth(0.8)

    fig.text(0.5, 0.03, f"Van holdout, top-1 %, one seed (spread 0.33 pp) · axis starts at {AXIS_START} "
             "· ties: crop scale, colour jitter, head lr 30×", ha="center", va="bottom",
             fontsize=11, color=INK_2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=PAGE)
        print(f"[plot] wrote {out.with_suffix('.' + ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
