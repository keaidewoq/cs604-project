"""Slide figure: top-k accuracy on the all-class holdout.

LoRA r24 no extras (lora_20261026_123804), 10,000 classes, 5 holdout photos each. Numbers from
report/progress_log.md (2026-10-26).

    python scripts/plot_topk.py --out figures/topk
writes <out>.png and <out>.svg.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_full_ft_ablation import INK, INK_2, PAGE, RECIPE, RULE

TOPK = {1: 77.8, 2: 86.0, 3: 89.2, 5: 92.0, 10: 94.4}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="figures/topk", help="output path without extension")
    args = p.parse_args()

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 13,
                         "text.color": INK, "xtick.color": INK_2, "ytick.color": INK})
    fill, edge = RECIPE
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=PAGE)
    ax.set_facecolor(PAGE)
    xs, accs = list(TOPK), list(TOPK.values())
    ax.plot(xs, accs, color=edge, lw=2, zorder=2)
    ax.scatter(xs, accs, s=90, color=fill, edgecolors=edge, linewidths=1.5, zorder=3)
    for x, acc in TOPK.items():
        below = x == 1  # the rising line would cross a label above this point
        ax.annotate(f"{acc:.1f}", (x, acc), xytext=(0, -13 if below else 12), textcoords="offset points",
                    ha="center", va="top" if below else "bottom", color=INK, fontsize=13, fontweight="bold")

    # gain over each step, under the middle of its segment
    for (x0, a0), (x1, a1) in zip(TOPK.items(), list(TOPK.items())[1:]):
        ax.annotate(f"+{a1 - a0:.1f} pts", ((x0 + x1) / 2, (a0 + a1) / 2), xytext=(10, -10),
                    textcoords="offset points", ha="left", va="top", color=edge, fontsize=12, fontweight="bold")

    ax.set_xticks(xs, [f"Top-{k}" for k in xs])
    ax.set_xlim(0.4, 10.6)
    ax.set_ylim(72, 100)
    ax.set_yticks(range(75, 101, 5))
    ax.set_ylabel("Accuracy (%)", color=INK_2, labelpad=8)
    ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0, pad=8)

    fig.text(0.01, 0.025, "Share of photos with the true species among the model's k highest-scoring guesses. "
             "Axis starts at 72.\nAll-class holdout (10,000 classes, 5 photos each), LoRA r24.",
             color=INK_2, fontsize=10, linespacing=1.35)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=PAGE)
        print(f"[plot] wrote {out.with_suffix('.' + ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
