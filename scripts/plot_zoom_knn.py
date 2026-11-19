"""Slide figures: what zoom and kNN re-ranking each add, and both together.

  zoom_knn_holdout     LoRA r24 no extras on the all-class holdout, crop 0.75, the only model with all
                       four setups measured (rerank_bakeoff, progress_log.md, 2026-10-27)
  zoom_only_holdout, knn_only_holdout   the same numbers, each method alone against Plain
  second_view_holdout  which second view to average in, 2 passes each, Van's full FT at 336 on Van's holdout
                       (origin/van-zoom progress_log.md, 2026-10-09; final_report_draft.md section 4.1)
  zoom_knn_codabench   Van's full FT at 448 on CodaBench; kNN alone was not scored on this model
                       (final_report_draft.md, single view 0.825, zoom 0.8322, zoom + kNN 0.8407)

All axes span the same number of points, so bar heights compare across the charts.

    python scripts/plot_zoom_knn.py --out_dir figures
writes <name>.png and <name>.svg for each.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from plot_full_ft_ablation import INK, INK_2, NEUTRAL, PAGE, RECIPE, RULE

HOLDOUT = [("Plain", 78.31), ("+ zoom", 79.04), ("+ kNN", 79.63), ("+ zoom and kNN", 80.27)]
CODABENCH = [("Plain", 82.50), ("+ zoom", 83.22), ("+ zoom and kNN", 84.07)]
VIEWS = [("Plain (1 pass)", 78.80), ("+ flip", 78.97), ("+ centred box", 79.43), ("+ aimed zoom", 79.59)]
BAR_W, AXIS_SPAN = 0.62, 4.2  # the span leaves room for the labels above the tallest bar


def figure(rows, axis_start, footnote, out: Path):
    axis_end = axis_start + AXIS_SPAN
    fig, ax = plt.subplots(figsize=(4 + 1.25 * len(rows), 5.6), facecolor=PAGE)
    ax.set_facecolor(PAGE)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(axis_start, axis_end)

    # round the bar ends by the same number of pixels in x and y
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    px_per_x = bbox.width / len(rows)
    px_per_y = bbox.height / AXIS_SPAN
    radius_x = 7 / px_per_x
    base = rows[0][1]
    for x, (_, acc) in enumerate(rows):
        fill, edge = NEUTRAL if acc == base else RECIPE
        ax.add_patch(FancyBboxPatch((x - BAR_W / 2, axis_start), BAR_W, acc - axis_start,
                                    boxstyle=f"round,pad=0,rounding_size={radius_x}",
                                    mutation_aspect=px_per_x / px_per_y,
                                    facecolor=fill, edgecolor=edge, linewidth=1.0, zorder=2))
        ax.text(x, acc + 0.08, f"{acc:.1f}", va="bottom", ha="center", color=INK, fontweight="bold",
                bbox={"facecolor": PAGE, "edgecolor": "none", "pad": 1.5})  # keeps gridlines off the text
        if acc != base:
            ax.text(x, acc + 0.33, f"{acc - base:+.1f} pts", va="bottom", ha="center", color=INK_2, fontsize=12,
                    bbox={"facecolor": PAGE, "edgecolor": "none", "pad": 1.5})

    ax.set_xticks(range(len(rows)), [r[0] for r in rows])
    ax.set_yticks(range(axis_start, int(axis_end) + 1))
    ax.set_ylabel("Top-1 accuracy (%)", color=INK_2, labelpad=8)
    ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=8)
    ax.tick_params(axis="x", colors=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.spines["bottom"].set_linewidth(0.8)

    fig.text(0.01, 0.025, footnote, color=INK_2, fontsize=10, linespacing=1.35)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=PAGE)
        print(f"[plot] wrote {out.with_suffix('.' + ext)}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="figures")
    args = p.parse_args()
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 13,
                         "text.color": INK, "xtick.color": INK_2, "ytick.color": INK})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figure(HOLDOUT, 77,
           "Axis starts at 77. Each gain is against Plain. All-class holdout (10,000 classes, 5 photos each),\n"
           "LoRA r24, centre crop 0.75. kNN settings chosen on one half of the classes and scored on the other.",
           out / "zoom_knn_holdout")
    plain, zoom, knn, _ = HOLDOUT
    figure([plain, zoom], 77,
           "Axis starts at 77. All-class holdout (10,000 classes,\n"
           "5 photos each), LoRA r24, centre crop 0.75.",
           out / "zoom_only_holdout")
    figure([plain, knn], 77,
           "Axis starts at 77. All-class holdout (10,000 classes, 5 photos each), LoRA r24,\n"
           "centre crop 0.75. kNN settings chosen on one half of the classes, scored on the other.",
           out / "knn_only_holdout")
    figure(VIEWS, 77,
           "Axis starts at 77. Each gain is against Plain, on the same model and photos. Van's holdout, full\n"
           "fine-tune at 336 px after 3 epochs, centre crop 0.75. Centred box: same size as the aimed box, placed centrally.",
           out / "second_view_holdout")
    figure(CODABENCH, 81,
           "Axis starts at 81. Each gain is against Plain. CodaBench validation, full fine-tune\n"
           "at 448 px, centre crop 0.9. kNN alone was not scored on this model.",
           out / "zoom_knn_codabench")


if __name__ == "__main__":
    main()
