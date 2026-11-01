"""Is model B really better than model A? Paired comparison on the same holdout photos, no training.

Both models are scored on the same labelled holdout, so each photo is one of: fixed (A wrong,
B right), broken (A right, B wrong), both right or both wrong. From those counts:

  - accuracy of each model with a 95% interval (Wilson, per photo);
  - the difference B - A with two 95% intervals from a paired bootstrap: resampling photos,
    and resampling classes (all 5 photos of a class together). Photos of one class are not
    independent (a hard class tends to get all 5 wrong), so the class interval is the honest
    one; the photo interval is shown for comparison;
  - McNemar's exact test: are fixes and breaks equally likely? (two-sided p-value);
  - are the fixes siblings? The same split as scripts/rerank_knn.py: A's wrong guess within 5
    class IDs of the truth is a sibling error; siblings' share of A's errors vs of the fixes.

The intervals cover which photos were drawn into the holdout, not training randomness: a
second seed of the same run also moves accuracy (~0.3 points here). Call a difference real
only if it clears both: the interval excludes 0 and the gap is beyond the seed spread.

Use checkpoints that never saw the holdout (pilot runs, or the all-class 40 fit / 5 holdout
runs), all scored on the same --pilot_root. Each model is scored once and cached as
preds_<name>.pt in --out_dir; delete the file to score again.

    python scripts/paired_compare.py --pilot_root <holdout data root> --out_dir <folder> \\
        --run r224=<run dir> --run r336=<run dir> [--baseline r224]

Every other run is compared with the baseline (default: the first --run). Writes
paired_<A>_vs_<B>.json, changes_<A>_vs_<B>.csv (every photo either model got wrong, with a
status column, for picking report examples) and paired_summary_<A>.csv (one row per
comparison). Running again with another --baseline into the same --out_dir reuses the cached
predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rerank_knn import fixes_by_distance

Z = 1.96
BOOT_CHUNK = 500


def load_or_score(name, run_dir, pilot_root, out, batch_size, workers, crop_pct):
    """-> dict of paths, labels (N,), top10 classes (N, 10), true-class rank (N,); cached per model."""
    cache = out / f"preds_{name}.pt"
    if cache.exists():
        p = torch.load(cache)
        print(f"[paired] {name}: cached {cache} ({p['run_dir']})")
        return p
    from scripts.error_analysis import score

    paths, labels, top_i, top_p, ranks = score(run_dir, pilot_root, batch_size, workers, crop_pct)
    p = {"run_dir": run_dir, "pilot_root": str(pilot_root), "crop_pct": crop_pct,
         "paths": paths, "labels": labels, "top_i": top_i, "top_p": top_p, "ranks": ranks}
    torch.save(p, cache)
    return p


def wilson(k, n):
    """95% interval of a proportion k / n."""
    p = k / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return centre - half, centre + half


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for b fixes and c breaks (binomial, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2)
            for k in range(min(b, c) + 1)]
    top = max(logs)
    return min(1.0, 2 * math.exp(top) * sum(math.exp(v - top) for v in logs))


def boot_photos(d, reps, gen):
    """Bootstrap the mean of d (per-photo B - A in {-1, 0, 1}): resampled counts are multinomial."""
    n = len(d)
    p_fix, p_break = float((d == 1).sum()) / n, float((d == -1).sum()) / n
    fixes = torch.binomial(torch.full((reps,), float(n)), torch.full((reps,), p_fix), generator=gen)
    p_rest = p_break / (1 - p_fix) if p_fix < 1 else 0.0
    breaks = torch.binomial(n - fixes, torch.full((reps,), p_rest), generator=gen)
    return (fixes - breaks) / n


def boot_classes(d, labels, reps, gen):
    """Bootstrap the mean of d resampling whole classes (cluster bootstrap)."""
    classes, inv = labels.unique(return_inverse=True)
    k = len(classes)
    d_sum = torch.zeros(k).index_add_(0, inv, d.float())
    n_sum = torch.zeros(k).index_add_(0, inv, torch.ones(len(d)))
    out = []
    for start in range(0, reps, BOOT_CHUNK):
        idx = torch.randint(k, (min(BOOT_CHUNK, reps - start), k), generator=gen)
        out.append(d_sum[idx].sum(1) / n_sum[idx].sum(1))
    return torch.cat(out)


def interval(samples):
    lo, hi = torch.quantile(samples, torch.tensor([0.025, 0.975])).tolist()
    return lo, hi


def pts(x):
    return round(100 * x, 2)


def compare(a, b, reps, seed):
    """Paired statistics of model b against baseline a (same photos, same order)."""
    if a["paths"] != b["paths"] or not torch.equal(a["labels"], b["labels"]):
        raise SystemExit("the two models were scored on different holdout photos: use the same --pilot_root")
    y = a["labels"]
    n = len(y)
    ok_a, ok_b = a["top_i"][:, 0] == y, b["top_i"][:, 0] == y
    fixed, broken = ~ok_a & ok_b, ok_a & ~ok_b
    both_wrong = ~ok_a & ~ok_b
    d = ok_b.long() - ok_a.long()
    gen = torch.Generator().manual_seed(seed)
    diff = float(d.float().mean())
    ci_photo = interval(boot_photos(d, reps, gen))
    ci_class = interval(boot_classes(d, y, reps, gen))
    acc = {}
    for name, ok in (("a", ok_a), ("b", ok_b)):
        lo, hi = wilson(int(ok.sum()), n)
        acc[name] = {"top1": pts(float(ok.float().mean())), "ci95": [pts(lo), pts(hi)]}
    same_guess = both_wrong & (a["top_i"][:, 0] == b["top_i"][:, 0])
    return {
        "photos": n,
        "classes": int(y.unique().numel()),
        "accuracy_a": acc["a"],
        "accuracy_b": acc["b"],
        "difference_points": pts(diff),
        "difference_ci95_by_class": [pts(v) for v in ci_class],
        "difference_ci95_by_photo": [pts(v) for v in ci_photo],
        "fixed": int(fixed.sum()),
        "broken": int(broken.sum()),
        "both_right": int((ok_a & ok_b).sum()),
        "both_wrong": int(both_wrong.sum()),
        "both_wrong_same_guess_share": round(float(same_guess.sum()) / max(int(both_wrong.sum()), 1), 4),
        "mcnemar_p": float(f"{mcnemar_exact(int(fixed.sum()), int(broken.sum())):.3g}"),
        # A's errors and how B changed them, split by class-ID distance (as in rerank_knn.py)
        "by_distance": fixes_by_distance(a["top_i"][:, 0], b["top_i"][:, 0], y, a["ranks"]),
    }


def write_changes(path, a, b):
    """One row per photo that either model got wrong."""
    y = a["labels"]
    ga, gb = a["top_i"][:, 0], b["top_i"][:, 0]
    keep = ((ga != y) | (gb != y)).nonzero().squeeze(1).tolist()
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "status", "guess_a", "guess_b", "truth_rank_a", "truth_rank_b",
                    "ids_a", "ids_b", "p_guess_a", "p_guess_b"])
        for i in keep:
            t, pa, pb = int(y[i]), int(ga[i]), int(gb[i])
            status = "fixed" if pa != t and pb == t else "broken" if pa == t else "both_wrong"
            w.writerow([a["paths"][i], t, status, pa, pb, int(a["ranks"][i]), int(b["ranks"][i]),
                        abs(pa - t), abs(pb - t), round(float(a["top_p"][i, 0]), 3), round(float(b["top_p"][i, 0]), 3)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot_root", required=True, help="data root with a labelled val.txt (the holdout)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR")
    p.add_argument("--baseline", default=None, help="run NAME the others are compared with (default: the first)")
    p.add_argument("--crop_pct", type=float, default=None, help="eval centre crop (default: the model's, 0.9)")
    p.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds = {}
    for spec in args.run:
        name, _, run_dir = spec.partition("=")
        preds[name] = load_or_score(name, run_dir, Path(args.pilot_root), out, args.batch_size, args.workers,
                                    args.crop_pct)
    base = args.baseline or next(iter(preds))
    if base not in preds:
        raise SystemExit(f"--baseline {base} is not one of the --run names {list(preds)}")

    rows = []
    for name, b in preds.items():
        if name == base:
            continue
        res = {"a": base, "b": name, "run_dir_a": preds[base]["run_dir"], "run_dir_b": b["run_dir"],
               **compare(preds[base], b, args.boot, args.seed)}
        (out / f"paired_{base}_vs_{name}.json").write_text(json.dumps(res, indent=2))
        write_changes(out / f"changes_{base}_vs_{name}.csv", preds[base], b)
        print(json.dumps(res, indent=2))
        dist = res["by_distance"]
        rows.append({"a": base, "b": name, "photos": res["photos"], "classes": res["classes"],
                     "top1_a": res["accuracy_a"]["top1"], "top1_a_ci95": res["accuracy_a"]["ci95"],
                     "top1_b": res["accuracy_b"]["top1"], "top1_b_ci95": res["accuracy_b"]["ci95"],
                     "diff_points": res["difference_points"],
                     "diff_ci95_by_class": res["difference_ci95_by_class"],
                     "diff_ci95_by_photo": res["difference_ci95_by_photo"],
                     "fixed": res["fixed"], "broken": res["broken"], "both_wrong": res["both_wrong"],
                     "mcnemar_p": res["mcnemar_p"],
                     "sibling_share_of_errors_a": dist["sibling_share_of_errors"],
                     "sibling_share_of_fixes": dist["sibling_share_of_fixes"],
                     "fix_rate_sibling_errors": dist["fix_rate_sibling_errors"],
                     "fix_rate_far_errors": dist["fix_rate_far_errors"]})
    with (out / f"paired_summary_{base}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[paired] wrote paired_*.json, changes_*.csv, paired_summary_{base}.csv in {out}")


if __name__ == "__main__":
    main()
