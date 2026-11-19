"""Slide figure: final results on Phase 1 and Phase 2, and what kNN re-ranking costs.

Phase 2 scores as reported by Anastasia (100,000 images); Phase 1, latency and bank numbers from
report/final_report_draft.md section 0 and report/progress_log.md (2026-10-27). The "limits" figures are
from the LoRA all-class holdout, not this model.

    python scripts/plot_test_results.py --out figures/test_results
writes <out>.png and <out>.svg.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from plot_full_ft_ablation import INK, INK_2, NEUTRAL, PAGE, RECIPE, RULE

COLUMNS = [(3.95, "Phase 1\n50,000 images"), (6.05, "Phase 2\n100,000 images"), (8.35, "Latency\nbatch 1"), (10.45, "Extra\nstorage")]
ROWS = [
    ("Zoom", ("83.22", "82.91", "23.43 ms", "None"), False),
    ("Zoom + kNN", ("84.07", "83.84", "Under 0.1 ms more", "0.7 GB bank"), True),
    ("kNN's gain", ("+0.85", "+0.93", "", ""), False),
]
GAINS = ["+0.93 on Phase 2, about 930 more correct photos", "No training and no trainable parameters",
         "Under 0.1 ms per photo once the bank exists", "Settings fixed in advance; the gain held on Phase 2"]
COSTS = ["A 0.7 GB bank must be kept and available", "Bank takes about 25–30 min to build at 448",
         "Rebuilt whenever the model changes", "Only fixes errors with the truth in the top 5 (about 64%*)",
         "Breaks about 1 right answer per 2 it fixes*"]


def panel(ax, x, y, w, h, style):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", facecolor=style[0],
                                edgecolor=style[1], linewidth=1.0, zorder=1))


def trade_offs(ax, x, top, w, h, title, items, style):
    panel(ax, x, top - h, w, h, style)
    ax.text(x + 0.2, top - 0.3, title, fontsize=14, fontweight="bold", va="center", zorder=2)
    for i, item in enumerate(items):
        ax.text(x + 0.2, top - 0.68 - i * 0.35, item, fontsize=12, va="center", zorder=2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="figures/test_results", help="output path without extension")
    args = p.parse_args()

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "text.color": INK})
    width, height = 12.0, 5.6
    fig = plt.figure(figsize=(width, height), facecolor=PAGE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    head = 5.2
    ax.text(0.3, head, "Full FT at 448 px", fontsize=12, color=INK_2, va="center")
    for x, label in COLUMNS:
        ax.text(x, head, label, fontsize=12, color=INK_2, va="center", ha="center", linespacing=1.25,
                fontweight="bold" if label.startswith("Phase 2") else "normal")
    ax.plot([0.25, 11.75], [head - 0.36, head - 0.36], color=RULE, linewidth=0.9)

    # highlight the Phase 2 column behind the rows
    row_h, y = 0.58, head - 0.36
    panel(ax, 5.2, y - 3 * row_h + 0.03, 1.7, 3 * row_h - 0.06, RECIPE)
    for name, values, best in ROWS:
        y -= row_h
        gain_row = name.startswith("kNN")
        ax.text(0.3, y + row_h / 2, name, fontsize=14, fontweight="bold" if best or gain_row else "normal", va="center", zorder=2)
        for (x, _), value in zip(COLUMNS, values):
            test = x == COLUMNS[1][0]
            ax.text(x, y + row_h / 2, value, fontsize=17 if test else 13, ha="center", va="center", zorder=2,
                    fontweight="bold" if test and not gain_row or gain_row else "normal")
        ax.plot([0.25, 11.75], [y, y], color=RULE, linewidth=0.5, zorder=0)

    top = y - 0.3
    panel_h, gap = 2.3, 0.3
    panel_w = (11.5 - gap) / 2
    trade_offs(ax, 0.25, top, panel_w, panel_h, "What kNN gives", GAINS, RECIPE)
    trade_offs(ax, 0.25 + panel_w + gap, top, panel_w, panel_h, "What kNN costs", COSTS, NEUTRAL)

    ax.text(0.25, 0.17, "*Measured on the LoRA all-class holdout, not this model. Latency: A100, per image.",
            fontsize=9.5, color=INK_2, va="center")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=PAGE)
    fig.savefig(out.with_suffix(".svg"), facecolor=PAGE)
    print(f"wrote {out}.png and .svg")


if __name__ == "__main__":
    main()
