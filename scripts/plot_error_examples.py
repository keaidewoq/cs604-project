"""Slide figure: four real mistakes, one for each way a photo goes wrong (presentation slide 13).

Hand-picked from errors.html, a 100-row tagging gallery of the full_proto_336_len3 screening run
(full FT at 336 px, crop 0.75). They illustrate kinds of error and say nothing about how common
each kind is: 60 of the gallery's rows were chosen for being confident or from confused pairs.
Probabilities, photos and class IDs are read from the gallery, species names from
outputs/inat21_class_names.csv.

    python scripts/plot_error_examples.py --html ~/Downloads/errors.html --out figures/error_examples
writes <out>.png and <out>.svg.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

PAGE, INK, INK_2 = "#fbfaf7", "#3d3d3a", "#5f5e5a"
TRUE_INK, PRED_INK = "#0f6e56", "#b5441f"

# (gallery row number, what the example shows, which 2 of the 3 true-class training photos to show:
# the subject is too small or dark to make out in the ones left out)
EXAMPLES = [
    (52, "Look-alike relatives", (2, 0)),
    (6, "Tiny subject", (0, 1)),
    (63, "Subject outside the crop", (2, 0)),
    (95, "Unfamiliar look", (0, 1)),
]
FIG_W, FIG_H = 12.0, 6.85
ROW_RE = re.compile(r'<section class="row" data-key="\d+">(.*?)</section>', re.S)
META_RE = re.compile(r"true <b>(\d+)</b> \(.*?\) &rarr; predicted <b>(\d+)</b> \(p=([\d.]+)\)", re.S)
IMAGE_RE = re.compile(r"data:image/jpeg;base64,([A-Za-z0-9+/=]+)")


def decode(data):
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def place(fig, image, x, y, w, h):
    """Draw image inside the box (inches, origin bottom-left), centred, keeping its proportions."""
    scale = min(w / image.width, h / image.height)
    iw, ih = image.width * scale, image.height * scale
    ax = fig.add_axes([(x + (w - iw) / 2) / FIG_W, (y + (h - ih) / 2) / FIG_H, iw / FIG_W, ih / FIG_H])
    ax.imshow(image, interpolation="lanczos")
    ax.axis("off")


def label(fig, x, y, text, width=20, **kw):
    # line by line on a fixed baseline, so neighbouring labels line up whatever their letters
    for i, line in enumerate(textwrap.wrap(text, width)):
        fig.text(x / FIG_W, (y - 0.14 - i * 0.19) / FIG_H, line, va="baseline", ha="left", **kw)


def species(names, class_id):
    row = names[class_id]
    name = row["common_name"] or row["name"]
    return name[0].upper() + name[1:]


def cell(fig, x, y, title, photos, true_picks, true_name, pred_name, prob):
    """One example; (x, y) is the top-left corner of the cell in inches."""
    fig.text(x / FIG_W, y / FIG_H, title, va="top", ha="left", fontsize=14, fontweight="bold")
    top = y - 0.33
    label(fig, x, top, "Holdout photo", fontsize=10.5, color=INK_2)
    label(fig, x + 2.55, top, f"True: {true_name}", fontsize=10.5, color=TRUE_INK, fontweight="bold")
    label(fig, x + 4.15, top, f"Model: {pred_name} ({prob:.0%})", fontsize=10.5, color=PRED_INK, fontweight="bold")

    image_top = top - 0.5
    query, true_images, pred_images = photos[0], [photos[1 + i] for i in true_picks], photos[4:7]
    place(fig, query, x, image_top - 2.0, 2.4, 2.0)
    for column, images in ((x + 2.55, true_images), (x + 4.15, pred_images)):
        for i, image in enumerate(images[:2]):
            place(fig, image, column, image_top - 0.98 - i * 1.02, 1.45, 0.98)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--html", required=True, help="the errors.html gallery")
    p.add_argument("--names", default="outputs/inat21_class_names.csv")
    p.add_argument("--out", default="figures/error_examples", help="output path without extension")
    args = p.parse_args()

    rows = ROW_RE.findall(Path(args.html).read_text(encoding="utf8"))
    with open(args.names, encoding="utf8") as f:
        names = {int(r["class"]): r for r in csv.DictReader(f)}

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "text.color": INK})
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=PAGE)
    corners = [(0.25, 6.7), (6.2, 6.7), (0.25, 3.5), (6.2, 3.5)]
    for (number, title, true_picks), (x, y) in zip(EXAMPLES, corners):
        html = rows[number - 1]
        true_id, pred_id, prob = META_RE.search(html).groups()
        photos = [decode(d) for d in IMAGE_RE.findall(html)]
        cell(fig, x, y, title, photos, true_picks, species(names, int(true_id)), species(names, int(pred_id)),
             float(prob))
    fig.text(0.25 / FIG_W, 0.08 / FIG_H,
             "Red box = the crop the model saw. Right columns = training photos of each class. % = model confidence.\n"
             "Hand-picked illustrations, not how common each kind is. Full FT at 336 px (screening run), not a final model.",
             va="bottom", ha="left", fontsize=9.5, color=INK_2, linespacing=1.25)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=PAGE)
    fig.savefig(out.with_suffix(".svg"), facecolor=PAGE)
    print(f"wrote {out}.png and .svg")


if __name__ == "__main__":
    main()
