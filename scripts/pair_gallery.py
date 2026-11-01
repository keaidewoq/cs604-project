"""Look at the classes the model mixes up: photos of both classes of each pair, side by side.
No model needed: reads confused_pairs.csv (scripts/confusion_pairs.py) and the errors CSV it
came from (scripts/error_analysis.py), plus the training photos under --data_root.

For each pair (class A, class B), two rows of --per_class photos:
  row A: first the holdout photos of A that the model called B (red frame, with the guess),
         then other photos of A;
  row B: the same for B.
Pairs are taken from the top of confused_pairs.csv (most confusions first); --mutual_only keeps
pairs confused in both directions, the clearest look-alikes. --pairs A-B adds pairs by hand.

Writes pairs_<page>.png (--pairs_per_page pairs each) and pairs_shown.csv to --out_dir.
--classes_only writes just classes.txt (the classes those pairs need) and needs no photos.

    python scripts/pair_gallery.py --pairs_csv <dir>/confused_pairs.csv \\
        --errors <dir>/errors_lora_all.csv --data_root /content/data --out_dir <folder>
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image


def read_pairs(path, top, mutual_only):
    pairs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if mutual_only and row["mutual"] != "1":
                continue
            pairs.append(row)
            if len(pairs) == top:
                break
    return pairs


def read_photos(data_root):
    """class -> photo paths (relative to data_root), from train.txt."""
    photos = defaultdict(list)
    for line in (Path(data_root) / "train.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            photos[int(parts[1])].append(parts[0])
    return photos


def draw_page(path, pairs, photos, wrong, data_root, per_class, size, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    fig, axes = plt.subplots(2 * len(pairs), per_class, squeeze=False,
                             figsize=(1.9 * per_class, 2.1 * 2 * len(pairs)))
    for i, row in enumerate(pairs):
        a, b = int(row["class_a"]), int(row["class_b"])
        for j, (c, other) in enumerate(((a, b), (b, a))):
            missed = wrong.get((c, other), [])[:per_class]
            rest = [p for p in photos.get(c, []) if p not in missed]
            rest = rng.sample(rest, min(len(rest), per_class - len(missed)))
            cells = [(p, True) for p in missed] + [(p, False) for p in rest]
            for k in range(per_class):
                ax = axes[2 * i + j, k]
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                if k >= len(cells):
                    continue
                p, is_wrong = cells[k]
                img = Image.open(Path(data_root) / p).convert("RGB")
                img.thumbnail((size, size))  # keep the aspect ratio: a stretched animal is harder to judge
                ax.imshow(img)
                if is_wrong:
                    for side in ax.spines.values():
                        side.set_visible(True)
                        side.set_color("red")
                        side.set_linewidth(3)
                    ax.set_xlabel(f"called {other}", fontsize=8, color="red")
            acc = row["accuracy_a"] if c == a else row["accuracy_b"]
            axes[2 * i + j, 0].set_ylabel(f"class {c}\nacc {float(acc):.0%}", fontsize=9)
        n_ab, n_ba = row["a_called_b"], row["b_called_a"]
        axes[2 * i, 0].set_title(f"pair {a} / {b}: {a} called {b} x{n_ab}, {b} called {a} x{n_ba}, "
                                 f"ID gap {row['id_gap']}", loc="left", fontsize=10)
    # Fixed margins: tight_layout would shrink every photo to make room for the long pair titles
    w, h = fig.get_size_inches()
    fig.subplots_adjust(left=0.55 / w, right=1 - 0.05 / w, top=1 - 0.45 / h, bottom=0.25 / h,
                        wspace=0.05, hspace=0.45)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs_csv", required=True, help="confused_pairs.csv from scripts/confusion_pairs.py")
    p.add_argument("--errors", required=True, help="errors CSV the pairs came from (scripts/error_analysis.py)")
    p.add_argument("--data_root", help="full data root with train.txt and train/")
    p.add_argument("--classes_only", action="store_true",
                   help="only write classes.txt (the classes the chosen pairs need), so just those can be fetched")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--top", type=int, default=30, help="pairs taken from the top of confused_pairs.csv")
    p.add_argument("--mutual_only", action="store_true", help="only pairs confused in both directions")
    p.add_argument("--pairs", nargs="*", default=[], metavar="A-B", help="extra pairs, e.g. 2099-2100")
    p.add_argument("--per_class", type=int, default=8, help="photos per class row")
    p.add_argument("--pairs_per_page", type=int, default=5)
    p.add_argument("--size", type=int, default=224, help="pixels per photo")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pairs = read_pairs(args.pairs_csv, args.top, args.mutual_only)
    by_key = {(int(r["class_a"]), int(r["class_b"])): r for r in read_pairs(args.pairs_csv, None, False)}
    for spec in args.pairs:
        a, b = sorted(int(v) for v in spec.split("-"))
        row = by_key.get((a, b), {"class_a": a, "class_b": b, "a_called_b": 0, "b_called_a": 0,
                                  "id_gap": b - a, "accuracy_a": "nan", "accuracy_b": "nan", "mutual": "0"})
        if row not in pairs:
            pairs.append(row)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.classes_only:
        classes = sorted({int(r[k]) for r in pairs for k in ("class_a", "class_b")})
        (out / "classes.txt").write_text("\n".join(map(str, classes)) + "\n")
        print(f"[pairs] {len(pairs)} pairs need {len(classes)} classes; wrote {out / 'classes.txt'}")
        return
    if not args.data_root:
        p.error("--data_root is required unless --classes_only")

    wrong = defaultdict(list)   # (true, guess) -> holdout photos
    with open(args.errors, newline="") as f:
        for row in csv.DictReader(f):
            wrong[(int(row["true"]), int(row["guess1"]))].append(row["path"])

    photos = read_photos(args.data_root)
    pages = [pairs[i:i + args.pairs_per_page] for i in range(0, len(pairs), args.pairs_per_page)]
    for n, page in enumerate(pages, 1):
        draw_page(out / f"pairs_{n:02d}.png", page, photos, wrong, args.data_root,
                  args.per_class, args.size, args.seed)
        print(f"[pairs] page {n}/{len(pages)}: " + ", ".join(f"{r['class_a']}/{r['class_b']}" for r in page))
    with (out / "pairs_shown.csv").open("w", newline="") as f:
        fields = ["class_a", "class_b", "a_called_b", "b_called_a", "id_gap", "accuracy_a", "accuracy_b", "mutual"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(pairs)
    print(f"[pairs] wrote {len(pages)} pages of {len(pairs)} pairs and pairs_shown.csv in {out}")


if __name__ == "__main__":
    main()
