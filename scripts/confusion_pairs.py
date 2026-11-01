"""Which classes does the model get wrong, and which classes does it mix them up with?
No model needed: reads an errors_<name>.csv from scripts/error_analysis.py.

The errors file lists only misclassified photos, so every holdout class is assumed to have
--per_class photos (5 on the all-class holdout c10000_h5_s0). Class IDs follow the iNat
taxonomy, so a small ID gap between two confused classes means close relatives.

Writes to --out_dir:
  - class_accuracy.csv: one row per class with errors: photos wrong out of --per_class,
    accuracy, the class it is most often mistaken for (top guess), how often, and the ID gap;
  - confused_pairs.csv: one row per pair of classes (either direction), with how often A was
    called B and B was called A. `mutual` pairs go both ways: the strongest sign of two
    look-alike classes rather than one bad class;
and prints a summary.

    python scripts/confusion_pairs.py \\
        --errors <out_dir>/errors_lora_all.csv --out_dir <folder>
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--errors", required=True, help="errors_<name>.csv from scripts/error_analysis.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--per_class", type=int, default=5, help="holdout photos per class")
    p.add_argument("--num_classes", type=int, default=10000, help="classes in the holdout (1000 for a pilot)")
    p.add_argument("--near", type=int, default=10, help="ID gap that counts as a close relative")
    args = p.parse_args()

    wrong = Counter()                  # true class -> photos wrong
    called = defaultdict(Counter)      # true class -> top guess -> count
    with open(args.errors, newline="") as f:
        for row in csv.DictReader(f):
            t, g = int(row["true"]), int(row["guess1"])
            wrong[t] += 1
            called[t][g] += 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def acc(c):
        return 1 - wrong[c] / args.per_class

    with (out / "class_accuracy.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "wrong", "accuracy", "most_called", "times", "id_gap"])
        for c in sorted(wrong, key=lambda c: (-wrong[c], c)):
            g, n = called[c].most_common(1)[0]
            w.writerow([c, wrong[c], round(acc(c), 2), g, n, abs(g - c)])

    pairs = {}
    for t, guesses in called.items():
        for g, n in guesses.items():
            a, b = min(t, g), max(t, g)
            ab, ba = pairs.get((a, b), (0, 0))
            pairs[(a, b)] = (ab + n, ba) if t == a else (ab, ba + n)
    rows = sorted(((a, b, ab, ba) for (a, b), (ab, ba) in pairs.items()),
                  key=lambda r: (-(r[2] + r[3]), -min(r[2], r[3]), r[0]))
    with (out / "confused_pairs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_a", "class_b", "a_called_b", "b_called_a", "total", "mutual",
                    "id_gap", "accuracy_a", "accuracy_b"])
        for a, b, ab, ba in rows:
            w.writerow([a, b, ab, ba, ab + ba, int(ab > 0 and ba > 0), b - a,
                        round(acc(a), 2), round(acc(b), 2)])

    errors = sum(wrong.values())
    near = sum(n for t, gs in called.items() for g, n in gs.items() if abs(g - t) <= args.near)
    hist = Counter(wrong.values())
    hist[0] = args.num_classes - len(wrong)  # holdout classes with no errors (their IDs need not be 0..N-1)
    summary = {
        "photos": args.per_class * args.num_classes,
        "errors": errors,
        "classes_by_photos_wrong": {f"{k}/{args.per_class}": hist[k] for k in sorted(hist)},
        f"errors_within_{args.near}_ids": round(near / errors, 3),
        "pairs_confused": len(rows),
        "mutual_pairs": sum(1 for r in rows if r[2] and r[3]),
        f"mutual_pairs_within_{args.near}_ids": sum(1 for r in rows if r[2] and r[3] and r[1] - r[0] <= args.near),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[pairs] wrote class_accuracy.csv, confused_pairs.csv, summary.json in {out}")


if __name__ == "__main__":
    main()
