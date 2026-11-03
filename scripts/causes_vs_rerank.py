"""Which errors does kNN re-ranking fix? Join its per-photo changes with the error causes. No GPU.

Reads rerank_changes.csv (scripts/rerank_knn.py) and causes_per_error.csv + classes.csv
(scripts/error_causes.py), both from the same checkpoint and holdout. For each cause: how many errors
have the flag, how many have the truth in the re-ranker's top 5 (it cannot fix the others), and how many
it fixed. Photos it broke were correct before, so they have no per-photo flags: breaks are only split by
messy class, which is a property of the class.

    python scripts/causes_vs_rerank.py --changes <rerank>/rerank_changes.csv --causes_dir <causes> \\
        --out <folder>/causes_vs_rerank.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CAUSES = ["sibling", "unseen_look", "two_looks", "messy_class"]
TOP_K = 5  # the re-ranker's candidates (K of rerank_knn.SUBMITTED)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def share(part, whole):
    return round(part / whole, 4) if whole else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--changes", required=True, help="rerank_changes.csv")
    p.add_argument("--causes_dir", required=True, help="folder with causes_per_error.csv and classes.csv")
    p.add_argument("--out", required=True, help="output json")
    args = p.parse_args()

    changes = {r["path"]: r for r in read_csv(args.changes)}
    all_errors = read_csv(Path(args.causes_dir) / "causes_per_error.csv")
    # the two runs make separate forward passes, so a few near-ties can flip; many means a wrong file
    errors = [e for e in all_errors if e["path"] in changes and e["guess"] == changes[e["path"]]["before"]]
    unmatched = len(all_errors) - len(errors)
    assert unmatched <= 0.01 * len(all_errors), (
        f"{unmatched} of {len(all_errors)} errors do not match {args.changes}: different checkpoint or holdout?")

    def row(group):
        fixed = sum(changes[e["path"]]["fixed"] == "1" for e in group)
        reachable = sum(int(changes[e["path"]]["truth_rank_before"]) <= TOP_K for e in group)
        return {"errors": len(group), "truth_in_top5": reachable, "fixed": fixed,
                "fix_rate": share(fixed, len(group)), "fix_rate_when_truth_in_top5": share(fixed, reachable)}

    by_flag = {c: row([e for e in errors if e[c] == "True"]) for c in CAUSES}
    by_flag["sibling_only"] = row([e for e in errors if e["sibling"] == "True" and e["unseen_look"] == "False"])
    by_flag["unseen_look_only"] = row([e for e in errors if e["unseen_look"] == "True" and e["sibling"] == "False"])
    by_flag["no_flag"] = row([e for e in errors if not any(e[c] == "True" for c in CAUSES)])

    messy = {r["class"] for r in read_csv(Path(args.causes_dir) / "classes.csv") if float(r["messy_share"]) >= 0.2}
    broken = [r for r in changes.values() if r["broken"] == "1"]
    fixed_total = sum(r["fixed"] == "1" for r in changes.values())
    holdout = len(read_csv(Path(args.causes_dir) / "classes.csv")) * 5
    summary = {"errors_before": len(all_errors), "errors_unmatched": unmatched, "fixed": fixed_total,
               "broken": len(broken), "net_points_on_all_photos": round(100 * (fixed_total - len(broken)) / holdout, 2),
               "all_errors": row(errors), "by_flag": by_flag,
               "broken_in_messy_classes": sum(r["label"] in messy for r in broken),
               "fixed_in_messy_classes": by_flag["messy_class"]["fixed"]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
