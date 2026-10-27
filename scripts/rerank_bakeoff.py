"""Which re-ranker is best? A fair comparison on a labelled holdout. No training of the model.

Uses a checkpoint trained on <pilot_root>/train.txt and its labelled holdout <pilot_root>/val.txt
(e.g. the all-class holdout run: 40 photos per class trained, 5 held out, all 10,000 classes).
The bank is the training photos' features from the same checkpoint. For each holdout photo the
model's top K = 5 classes are the candidates (from the plain view, and with --zoom also from
Van's zoom probabilities), and every re-ranker picks one of them:

  formula re-rankers   score = log p + beta * s(c), one beta chosen per re-ranker
    knn_mean3          s = mean of the 3 best cosines to c's training photos (the current method)
    knn_max            s = the best cosine
    knn_meanall        s = mean cosine to all of c's training photos
    ncm                s = cosine to c's mean training feature
    centres3           s = best cosine to 3 k-means centres of c's training photos
    vote               s = log sum over the photo's 50 nearest bank photos of class c of
                           exp(cosine / 0.05) (weighted kNN vote over the whole bank)
  learned re-rankers   trained on the signals of every candidate (log p, rank, gap to the top
                       candidate, the six s above, class spread, sibling flag, other view's log p)
    linear             softmax over the K candidates of a linear score (a learned formula)
    trees              gradient-boosted trees, one row per candidate (scikit-learn)
  + balancing          each of the above followed by Van's class-count balancing
                       (src/van_balance.py, Sinkhorn to the holdout's 5 photos per class), and
                       balancing alone (Van's way, top-100 probabilities)

Fair protocol for all: holdout classes split in two halves; beta (or the learned model) is chosen
on one half and scored on the other, then swapped. Reported: held-out accuracy, gain over the
model's own top-1, fixes (wrong -> right) and breaks (right -> wrong). Writes bakeoff.csv,
bakeoff.json and linear_reranker.json (the linear re-ranker refit on the whole holdout, with its
feature weights, for applying to other checkpoints).

    python scripts/rerank_bakeoff.py --run_dir <run> --pilot_root /content/pilot_data/c10000_h5_s0 \\
        --out_dir <folder> [--zoom --crop_pct 0.9]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.predict_rerank import build_bank, class_index, features_and_logits
from src import van_balance
from src.data import ImageListDataset, build_transforms
from src.model import build_model
from src.predict import load_checkpoint
from src.utils import apply_overrides, get_device, load_config, set_seed

K = 5
TOPN = 100           # probabilities kept per photo (as Van's balancing)
VOTE_NEIGHBOURS, VOTE_TAU = 50, 0.05
BETAS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
VOTE_BETAS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
FORMULAS = ("knn_mean3", "knn_max", "knn_meanall", "ncm", "centres3", "vote")
SIGNALS = ("lp", "rank", "gap") + FORMULAS + ("spread", "sibling", "lp_other")


@torch.no_grad()
def holdout_pass(model, ds, zoom, mass, bs, nw, device, use_bf16):
    """Plain features, top-N plain log-probs and (with zoom) top-N zoom log-probs, per photo."""
    from src import van_zoom

    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=device.type == "cuda")
    feats, labels, plain_v, plain_i, zoom_v, zoom_i = [], [], [], [], [], []
    for batch, y, _ in tqdm(loader, desc="holdout"):
        small = (batch[0] if zoom else batch).to(device, non_blocking=True)
        f, logits, tokens = features_and_logits(model, small, use_bf16)
        lp = logits.log_softmax(1)
        v, i = lp.topk(min(TOPN, lp.shape[1]), dim=1)
        plain_v.append(v.cpu()), plain_i.append(i.cpu())
        if zoom:
            big = batch[1].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                zoomed = van_zoom.zoom_view(model, tokens, logits, big, mass=mass)
            _, logits_z, _ = features_and_logits(model, zoomed, use_bf16)
            lpz = ((logits.softmax(1) + logits_z.softmax(1)) / 2).clamp_min(1e-12).log()
            v, i = lpz.topk(min(TOPN, lpz.shape[1]), dim=1)
            zoom_v.append(v.cpu()), zoom_i.append(i.cpu())
        feats.append(f.half().cpu())
        labels.append(y)
    out = {"f": torch.cat(feats), "y": torch.cat(labels), "plain": (torch.cat(plain_v), torch.cat(plain_i))}
    if zoom:
        out["zoom"] = (torch.cat(zoom_v), torch.cat(zoom_i))
    return out


def lookup(vals, ids, cand):
    """log-prob of each candidate from a top-N list; missing -> (smallest kept) - 1."""
    hit = ids[:, None, :] == cand[:, :, None]                          # (N, K, TOPN)
    floor = vals[:, -1:].expand(-1, cand.shape[1]) - 1.0
    got = (vals[:, None, :] * hit).sum(-1)
    return torch.where(hit.any(-1), got, floor)


@torch.no_grad()
def class_stats(bank, idx, device):
    """Per class: mean feature, 3 k-means centres, spread (1 - mean cosine to the mean)."""
    C, L = idx.shape
    bank_d = bank.to(device)  # index and bank on the same device
    means = torch.zeros(C, bank.shape[1])
    centres = torch.zeros(C, 3, bank.shape[1])
    spread = torch.zeros(C)
    for s in range(0, C, 1024):
        rows = idx[s:s + 1024].to(device)
        valid = rows >= 0
        x = bank_d[rows.clamp_min(0)].float() * valid[..., None]        # (b, L, D)
        n = valid.sum(1, keepdim=True).clamp_min(1)
        mu = F.normalize(x.sum(1) / n, dim=-1)
        sim = (x * mu[:, None]).sum(-1)
        spread[s:s + 1024] = (1 - (sim * valid).sum(1) / n.squeeze(1)).cpu()
        # 3-means per class, initialised at the photo closest to the mean and the two farthest
        far = sim.masked_fill(~valid, 2).argsort(1)[:, :2]
        ar = torch.arange(len(x), device=x.device)
        cen = torch.stack([x[ar, sim.masked_fill(~valid, -2).argmax(1)], x[ar, far[:, 0]], x[ar, far[:, 1]]], 1)
        for _ in range(10):
            assign = torch.einsum("bld,bkd->blk", x, cen).argmax(-1).masked_fill(~valid, -1)
            for k in range(3):
                mk = (assign == k).float()[..., None]
                cen[:, k] = torch.where(mk.sum(1) > 0, (x * mk).sum(1) / mk.sum(1).clamp_min(1), cen[:, k])
        means[s:s + 1024] = mu.cpu()
        centres[s:s + 1024] = F.normalize(cen, dim=-1).cpu()
    return means, centres, spread


@torch.no_grad()
def candidate_signals(f, cand, lp_base, lp_other, bank, bank_y, idx, means, centres, spread, device):
    """(N, K, len(SIGNALS)) signals for every candidate."""
    N = len(f)
    out = torch.zeros(N, K, len(SIGNALS))
    bank_f = bank.to(device).float()  # once, not per chunk
    idx_d, means_d, centres_d, spread_d, bank_y_d = (t.to(device) for t in (idx, means, centres, spread, bank_y))
    for s in tqdm(range(0, N, 512), desc="signals"):
        fb = f[s:s + 512].to(device).float()
        cb = cand[s:s + 512].to(device)
        rows = idx_d[cb]
        valid = rows >= 0
        sims = torch.einsum("bd,bkld->bkl", fb, bank_f[rows.clamp_min(0)])
        top = sims.masked_fill(~valid, -2).sort(-1, descending=True).values
        knn_max = top[..., 0]
        knn_mean3 = top[..., :3].clamp_min(-1).mean(-1)
        knn_meanall = (sims * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        ncm = torch.einsum("bd,bkd->bk", fb, means_d[cb])
        cen = torch.einsum("bd,bkjd->bkj", fb, centres_d[cb]).max(-1).values
        # weighted vote over the photo's nearest bank photos (whole bank)
        allsim = fb @ bank_f.T
        nv, ni = allsim.topk(min(VOTE_NEIGHBOURS, allsim.shape[1]), dim=1)
        ny = bank_y_d[ni]
        w = (nv / VOTE_TAU).exp()
        vote = ((ny[:, None, :] == cb[:, :, None]) * w[:, None, :]).sum(-1)
        vote = (vote / w.sum(1, keepdim=True)).clamp_min(1e-6).log()
        lpb = lp_base[s:s + 512].to(device)
        sib = ((cb - cb[:, :1]).abs() <= 5).float()
        sib[:, 0] = 0
        cols = {"lp": lpb, "rank": torch.arange(K, device=device).float().expand_as(lpb),
                "gap": lpb[:, :1] - lpb, "knn_mean3": knn_mean3, "knn_max": knn_max, "knn_meanall": knn_meanall,
                "ncm": ncm, "centres3": cen, "vote": vote, "spread": spread_d[cb], "sibling": sib,
                "lp_other": lp_other[s:s + 512].to(device)}
        out[s:s + 512] = torch.stack([cols[k] for k in SIGNALS], -1).cpu()
    return out


def fit_linear(X, target, steps=300):
    """Conditional logit: softmax over candidates of X @ w. X (n, K, d) standardised; target (n,)."""
    w = torch.zeros(X.shape[-1], requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ w, target) + 1e-3 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--pilot_root", required=True, help="data root whose train.txt the run trained on; val.txt labelled")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--zoom", action="store_true")
    p.add_argument("--crop_pct", type=float, default=None)
    p.add_argument("--per_class", type=float, default=None, help="holdout photos per class for balancing (default: counted)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()
    t_start = time.time()

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    apply_overrides(cfg, data_root=args.pilot_root, batch_size=args.batch_size, num_workers=args.workers)
    device = get_device()
    set_seed(int(cfg.seed))
    model = build_model(cfg).to(device)
    load_checkpoint(model, run_dir / "last.pt", device)
    model.eval()
    use_bf16 = bool(cfg.bf16) and device.type == "cuda"
    bs, nw = int(cfg.batch_size), int(cfg.num_workers)
    _, bank_tf = build_transforms(cfg)
    bank, bank_y = build_bank(model, args.pilot_root, bank_tf, bs, nw, device, use_bf16)
    C = int(cfg.num_classes)
    idx = class_index(bank_y, C)
    if args.zoom:
        from src import van_zoom

        tf = van_zoom.two_view_transform(cfg, args.crop_pct)
        mass = van_zoom.DEFAULT_MASS
    else:
        _, tf = build_transforms(cfg, eval_crop_pct=args.crop_pct)
        mass = None
    H = holdout_pass(model, ImageListDataset(args.pilot_root, "val", tf), args.zoom, mass, bs, nw, device, use_bf16)
    y = H["y"]
    if not bool((y >= 0).all()):
        raise SystemExit("val.txt must be labelled (a pilot holdout)")
    means, centres, spread = class_stats(bank, idx, device)
    classes = sorted(set(y.tolist()))
    halfmap = {c: i % 2 for i, c in enumerate(classes)}
    half = torch.tensor([halfmap[c] for c in y.tolist()])
    per_class = args.per_class or float(len(y)) / len(classes)

    results, linear_out = [], {}
    for base in (["plain", "zoom"] if args.zoom else ["plain"]):
        vals, ids = H[base]
        other = H["zoom" if base == "plain" else "plain"] if args.zoom else H["plain"]
        cand, lp_base = ids[:, :K], vals[:, :K]
        lp_other = lookup(other[0], other[1], cand)
        X = candidate_signals(H["f"], cand, lp_base, lp_other, bank, bank_y, idx, means, centres, spread, device)
        top1 = cand[:, 0]
        right0 = top1 == y
        target = (cand == y[:, None]).float().argmax(1)
        reachable = (cand == y[:, None]).any(1)
        col = {k: j for j, k in enumerate(SIGNALS)}

        def held_out(score_fn):
            """score_fn(tune_mask) -> (N, K) scores, fitted on tune_mask only; returns preds and per-half info."""
            preds = torch.empty_like(y)
            scores_all = torch.empty(len(y), K)
            info = []
            for tune, test in ((0, 1), (1, 0)):
                scores, chosen = score_fn(half == tune)
                sel = half == test
                pr = cand.gather(1, scores.argmax(1, keepdim=True)).squeeze(1)
                preds[sel] = pr[sel]
                scores_all[sel] = scores[sel]
                info.append(chosen)
            return preds, scores_all, info

        def formula(name, betas):
            def fn(mask):
                best = max(betas, key=lambda b: float((cand.gather(1, (X[..., col["lp"]] + b * X[..., col[name]]).argmax(1, keepdim=True)).squeeze(1) == y)[mask].float().mean()))
                return X[..., col["lp"]] + best * X[..., col[name]], {"beta": best}
            return fn

        def linear(mask):
            m = mask & reachable
            mu, sd = X[mask].reshape(-1, X.shape[-1]).mean(0), X[mask].reshape(-1, X.shape[-1]).std(0).clamp_min(1e-6)
            w = fit_linear((X[m] - mu) / sd, target[m])
            return ((X - mu) / sd) @ w, {"weights": {k: round(float(w[j]), 4) for k, j in col.items()}}

        def trees(mask):
            from sklearn.ensemble import HistGradientBoostingClassifier

            m = mask & reachable
            feats = X[m].reshape(-1, X.shape[-1]).numpy()
            lab = (cand[m] == y[m][:, None]).reshape(-1).numpy()
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                                 l2_regularization=1.0, random_state=0)
            clf.fit(feats, lab)
            prob = torch.tensor(clf.predict_proba(X.reshape(-1, X.shape[-1]).numpy())[:, 1]).float().reshape(len(y), K)
            return prob.clamp(1e-6, 1).log(), {"trees": "200 iters, 31 leaves"}

        methods = {"model top-1": lambda mask: (X[..., col["lp"]], {})}
        for name in FORMULAS:
            methods[name] = formula(name, VOTE_BETAS if name == "vote" else BETAS)
        methods["learned linear"] = linear
        methods["learned trees"] = trees

        for name, fn in methods.items():
            preds, scores, info = held_out(fn)
            right = preds == y
            row = {"base": base, "method": name}
            for h in (0, 1):
                row[f"acc_half{h}"] = round(float(right[half == h].float().mean()), 4)
            row["acc"] = round(float(right.float().mean()), 4)
            row["gain_points"] = round(100 * float(right.float().mean() - right0.float().mean()), 2)
            row["fixes"] = int((right & ~right0).sum())
            row["breaks"] = int((~right & right0).sum())
            # + balancing: Sinkhorn on the re-ranker's candidate distribution, all holdout photos at once
            pb, _ = van_balance.sinkhorn(scores.softmax(1), cand, C, per_class)
            rb = pb == y
            row["acc_balanced"] = round(float(rb.float().mean()), 4)
            row["gain_balanced_points"] = round(100 * float(rb.float().mean() - right0.float().mean()), 2)
            row["chosen"] = json.dumps(info)
            results.append(row)
            print(f"[bakeoff] {base:5s} {name:15s} acc {row['acc']:.4f} ({row['gain_points']:+.2f})  "
                  f"+balancing {row['acc_balanced']:.4f} ({row['gain_balanced_points']:+.2f})  "
                  f"fixes {row['fixes']} breaks {row['breaks']}")
        # balancing alone, Van's way (top-100 probabilities)
        pb, _ = van_balance.sinkhorn(vals.exp(), ids, C, per_class)
        acc_b = float((pb == y).float().mean())
        results.append({"base": base, "method": "balancing alone (Van, top-100)", "acc": round(acc_b, 4),
                        "gain_points": round(100 * (acc_b - float(right0.float().mean())), 2)})
        print(f"[bakeoff] {base:5s} balancing alone  acc {acc_b:.4f}")
        # linear re-ranker refit on the whole holdout, for applying elsewhere
        mu, sd = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0).clamp_min(1e-6)
        w = fit_linear((X[reachable] - mu) / sd, target[reachable])
        linear_out[base] = {"signals": list(SIGNALS), "mean": mu.tolist(), "std": sd.tolist(), "weights": w.tolist(),
                            "K": K, "vote_neighbours": VOTE_NEIGHBOURS, "vote_tau": VOTE_TAU}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["base", "method", "acc", "gain_points", "acc_balanced", "gain_balanced_points", "fixes", "breaks",
              "acc_half0", "acc_half1", "chosen"]
    with (out / "bakeoff.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    (out / "bakeoff.json").write_text(json.dumps({"run_dir": str(run_dir), "pilot_root": args.pilot_root,
                                                  "holdout_photos": len(y), "per_class": per_class, "K": K,
                                                  "results": results, "minutes": round((time.time() - t_start) / 60, 1)},
                                                 indent=2))
    (out / "linear_reranker.json").write_text(json.dumps(linear_out, indent=2))
    print(f"[bakeoff] wrote bakeoff.csv, bakeoff.json, linear_reranker.json in {out}")


if __name__ == "__main__":
    main()
