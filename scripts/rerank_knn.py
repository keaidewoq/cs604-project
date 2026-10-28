"""Re-rank the model's top-K guesses with its nearest training photos. No training.

The classifier compares a photo with one prototype per class (the class average). Re-ranking
also compares it with the class's individual training photos: for each of the top-K
candidate classes c,

    score(c) = log p(c)  +  beta * knn(c),    knn(c) = mean of the m best cosines between the
                                                       photo and c's training photos

so a photo that closely matches one training photo of a class (an adult moth among mostly
caterpillar photos, a near-identical sibling) can move up. Only the pilot's fit photos are
used as the photo bank, never the holdout.

Honest evaluation on the pilot holdout (labelled, never trained on): the settings (K, m,
beta) are chosen on one half of the holdout classes and scored on the other half, then the
halves swap; the reported gain is the average of the two held-out halves. Fixes (wrong ->
right) and breaks (right -> wrong) are counted separately. Also reports top-1/2/3/5 and where
the true class sits when wrong (rank 2 vs 3-5 vs later).

Are the fixes siblings? For the submitted setting (K = 5, m = 3, beta = 20) the fixes and
breaks are split by class-ID distance between the guess and the truth: a sibling error's
original guess is within 5 IDs of the truth. If siblings are a larger share of the fixes than
of the errors, the re-ranking corrects siblings more than other errors.

    python scripts/rerank_knn.py --run_dir <pilot run> --pilot_root <pilot data root> --out_dir <folder>
writes rerank.json, rerank_grid.csv, fixes_by_distance.json and rerank_changes.csv (one row per
holdout photo that was wrong before or changed by re-ranking).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.error_causes import embed
from src.data import ImageListDataset, build_transforms
from src.model import build_model
from src.predict import load_checkpoint
from src.utils import apply_overrides, get_device, load_config, set_seed

KS = (2, 3, 5)
MS = (1, 3)
BETAS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
SUBMITTED = (5, 3, 20.0)      # K, m, beta of every CodaBench re-ranking file
SIBLING_IDS = 5               # an error is a sibling error if the guess is within this many IDs
DISTANCES = (1, 5, 10, 50)    # also reported, to show the trend


@torch.no_grad()
def holdout_logits(model, ds, batch_size, workers, device, use_bf16, k=10):
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    top_i, top_lp = [], []
    for images, _, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(images.to(device, non_blocking=True))
        lp = logits.float().log_softmax(1)
        v, i = lp.topk(k, dim=1)
        top_i.append(i.cpu())
        top_lp.append(v.cpu())
    return torch.cat(top_i), torch.cat(top_lp)


def knn_scores(hold_f, cand, bank, bank_idx, m):
    """(N, K) mean of the m best cosines between each photo and each candidate's training photos."""
    out = torch.full(cand.shape, -1.0)
    for n in range(cand.shape[0]):
        for k, c in enumerate(cand[n].tolist()):
            rows = bank_idx.get(c)
            if rows is None:
                continue
            sims = bank[rows] @ hold_f[n]
            out[n, k] = sims.topk(min(m, len(rows))).values.mean()
    return out


def rerank(cand, lp, knn, beta):
    """New top-1 class per photo."""
    score = lp + beta * knn
    return cand.gather(1, score.argmax(1, keepdim=True)).squeeze(1)


def share(part, whole):
    return round(float(part.sum()) / max(int(whole.sum()), 1), 4)


def fixes_by_distance(before, after, labels, rank):
    """Split errors, fixes and breaks of one re-ranking by class-ID distance from the truth.

    before / after: top-1 class per photo without / with re-ranking; rank: where the truth sat
    in the model's list before re-ranking (1 = correct, 99 = beyond the top 10)."""
    wrong0, right1 = before != labels, after == labels
    fixed, broken = wrong0 & right1, ~wrong0 & ~right1
    d0, d1 = (before - labels).abs(), (after - labels).abs()
    sib0 = wrong0 & (d0 <= SIBLING_IDS)
    far0 = wrong0 & (d0 > SIBLING_IDS)
    return {
        "sibling_ids": SIBLING_IDS,
        "errors_before": int(wrong0.sum()),
        "fixes": int(fixed.sum()),
        "breaks": int(broken.sum()),
        # the comparison: siblings' share of the errors vs their share of the fixes
        "sibling_share_of_errors": share(sib0, wrong0),
        "sibling_share_of_fixes": share(fixed & sib0, fixed),
        # how often each kind of error gets fixed
        "fix_rate_sibling_errors": share(fixed & sib0, sib0),
        "fix_rate_far_errors": share(fixed & far0, far0),
        # a break that lands on a sibling swaps one look-alike for another
        "breaks_onto_sibling_share": share(broken & (d1 <= SIBLING_IDS), broken),
        "fixes_by_truth_rank": {"2": int((fixed & (rank == 2)).sum()),
                                "3-5": int((fixed & (rank >= 3) & (rank <= 5)).sum())},
        "within_ids": {str(d): {"share_of_errors": share(wrong0 & (d0 <= d), wrong0),
                                "share_of_fixes": share(fixed & (d0 <= d), fixed)} for d in DISTANCES},
    }


def write_changes(path, paths, labels, before, after, rank):
    """One row per photo that was wrong before re-ranking or whose prediction changed."""
    keep = ((before != labels) | (before != after)).nonzero().squeeze(1).tolist()
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "before", "after", "truth_rank_before", "ids_before", "ids_after",
                    "fixed", "broken"])
        for i in keep:
            y, b, a = int(labels[i]), int(before[i]), int(after[i])
            w.writerow([paths[i], y, b, a, int(rank[i]), abs(b - y), abs(a - y),
                        int(b != y and a == y), int(b == y and a != y)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--pilot_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(Path(args.run_dir) / "config.yaml")
    apply_overrides(cfg, data_root=args.pilot_root, batch_size=args.batch_size, num_workers=args.workers)
    device = get_device()
    set_seed(int(cfg.seed))  # as src.train does: same frozen weights if the backbone is not pretrained
    model = build_model(cfg).to(device)
    load_checkpoint(model, Path(args.run_dir) / "last.pt", device)
    model.eval()
    _, eval_tf = build_transforms(cfg)
    use_bf16 = bool(cfg.bf16) and device.type == "cuda"
    bs, nw = int(cfg.batch_size), int(cfg.num_workers)

    bank, _, bank_y, _ = embed(model, ImageListDataset(args.pilot_root, "train", eval_tf), bs, nw, device, use_bf16)
    hold_ds = ImageListDataset(args.pilot_root, "val", eval_tf)
    hold_f, _, hold_y, hold_paths = embed(model, hold_ds, bs, nw, device, use_bf16)
    cand_all, lp_all = holdout_logits(model, hold_ds, bs, nw, device, use_bf16)
    bank_idx = defaultdict(list)
    for i, c in enumerate(bank_y.tolist()):
        bank_idx[c].append(i)
    bank_idx = {c: torch.tensor(v) for c, v in bank_idx.items()}

    # where the truth sits (top-10 list) before re-ranking
    hit = cand_all == hold_y[:, None]
    rank = torch.where(hit.any(1), hit.float().argmax(1) + 1, torch.full_like(hold_y, 99))
    wrong = rank > 1
    base = {
        "top1": round(float((rank <= 1).float().mean()), 4), "top2": round(float((rank <= 2).float().mean()), 4),
        "top3": round(float((rank <= 3).float().mean()), 4), "top5": round(float((rank <= 5).float().mean()), 4),
        "errors": int(wrong.sum()),
        "errors_truth_rank2": round(float((rank == 2).sum() / wrong.sum().clamp_min(1)), 4),
        "errors_truth_rank3to5": round(float(((rank >= 3) & (rank <= 5)).sum() / wrong.sum().clamp_min(1)), 4),
        "errors_truth_beyond5": round(float((rank > 5).sum() / wrong.sum().clamp_min(1)), 4),
    }

    full = {m: knn_scores(hold_f, cand_all[:, :max(KS)], bank, bank_idx, m) for m in MS}
    knn = {(k, m): full[m][:, :k] for k in KS for m in MS}
    classes = sorted(set(hold_y.tolist()))
    half = {c: i % 2 for i, c in enumerate(classes)}  # split by class, alternating
    fold = torch.tensor([half[c] for c in hold_y.tolist()])
    correct0 = cand_all[:, 0] == hold_y

    grid, by_fold = [], {0: [], 1: []}
    for k, m, beta in itertools.product(KS, MS, BETAS):
        new = rerank(cand_all[:, :k], lp_all[:, :k], knn[(k, m)], beta)
        ok = new == hold_y
        row = {"K": k, "m": m, "beta": beta}
        for f in (0, 1):
            sel = fold == f
            row[f"acc_half{f}"] = round(float(ok[sel].float().mean()), 4)
            by_fold[f].append((row[f"acc_half{f}"], k, m, beta))
        row["acc_all"] = round(float(ok.float().mean()), 4)
        row["fixes"] = int((ok & ~correct0).sum())
        row["breaks"] = int((~ok & correct0).sum())
        grid.append(row)

    # choose on one half, score on the other
    cross = []
    for tune, test in ((0, 1), (1, 0)):
        best = max(by_fold[tune], key=lambda t: (t[0], -t[3]))  # ties -> smaller beta
        _, k, m, beta = best
        new = rerank(cand_all[:, :k], lp_all[:, :k], knn[(k, m)], beta)
        sel = fold == test
        ok = new == hold_y
        cross.append({"tuned_on_half": tune, "K": k, "m": m, "beta": beta,
                      "test_half_before": round(float(correct0[sel].float().mean()), 4),
                      "test_half_after": round(float(ok[sel].float().mean()), 4),
                      "fixes": int((ok & ~correct0 & sel).sum()), "breaks": int((~ok & correct0 & sel).sum())})
    gain = sum(c["test_half_after"] - c["test_half_before"] for c in cross) / 2
    summary = {"run_dir": args.run_dir, "pilot_root": args.pilot_root, "holdout_photos": len(hold_y),
               "before": base, "cross_validated": cross,
               "held_out_gain_points": round(100 * gain, 2),
               "best_on_all_photos_optimistic": max(grid, key=lambda r: r["acc_all"])}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rerank.json").write_text(json.dumps(summary, indent=2))
    with (out / "rerank_grid.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0]))
        w.writeheader()
        w.writerows(grid)
    print(json.dumps(summary, indent=2))

    # the submitted setting on the whole holdout: are its fixes siblings?
    k, m, beta = SUBMITTED
    after = rerank(cand_all[:, :k], lp_all[:, :k], knn[(k, m)], beta)
    by_dist = {"K": k, "m": m, "beta": beta, **fixes_by_distance(cand_all[:, 0], after, hold_y, rank)}
    (out / "fixes_by_distance.json").write_text(json.dumps(by_dist, indent=2))
    write_changes(out / "rerank_changes.csv", hold_paths, hold_y, cand_all[:, 0], after, rank)
    print(json.dumps(by_dist, indent=2))


if __name__ == "__main__":
    main()
