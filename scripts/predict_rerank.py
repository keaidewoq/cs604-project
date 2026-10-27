"""CodaBench submission with kNN re-ranking of the model's top-K guesses, optionally on top of zoom.

For each photo, the model's top K classes (by log-probability) are re-scored with how close
the photo is to each candidate's own training photos:

    score(c) = log p(c) + beta * knn(c),   knn(c) = mean of the m best cosines between the photo's
                                                    feature and c's training-photo features

and the best-scoring candidate is the prediction. Defaults K = 5, m = 3, beta = 20 were chosen
on held-out halves of the all-class holdout (+1.29 points there; CodaBench +1.40 on the LoRA
r24 no-extras checkpoint; report/progress_log.md, 2026-10-26/27). The bank is every photo in
<data_root>/train.txt, embedded once by the same checkpoint (plain view); --bank caches it.

--zoom reproduces Van's zoom prediction (src.predict --zoom: the plain view's softmax plus the
softmax of a zoomed view of the region the model used, src/van_zoom.py; with --crop_pct 0.9
that is her full FT 448 setting, 0.832) and re-ranks those zoom probabilities; the kNN feature
is the plain view's, matching the bank.

One pass writes every variant, so they compare on the same photos:
  <split>_team_<team>_plain.txt         the model's own top-1 (plain view)
  <split>_team_<team>_knn_b<beta>.txt   plain + kNN, for each --betas value
  with --zoom also
  <split>_team_<team>_zoom.txt          zoom only (Van's method)
  <split>_team_<team>_zoom_knn_b<beta>.txt
and <split>_team_<team>.txt = the main submission: zoom + kNN (or plain + kNN without --zoom)
at the first --betas value. If the split is labelled (a pilot holdout as data_root), every
variant is scored, and beta is chosen on half of the classes and tested on the other half.

    python scripts/predict_rerank.py --run_dir <run> --data_root /content/data --split val \\
        --team 08 --out_dir <folder> --bank <folder>/bank.pt [--zoom --crop_pct 0.9] [--betas 20 10 40]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ImageListDataset, build_transforms
from src.model import build_model
from src.predict import load_checkpoint
from src.utils import apply_overrides, get_device, load_config, set_seed


def features_and_logits(model, images, use_bf16, tokens=None):
    with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=use_bf16):
        if tokens is None:
            tokens = model.forward_features(images)
        f = model.forward_head(tokens, pre_logits=True)
        logits = model.get_classifier()(f)
    return F.normalize(f.float(), dim=-1), logits.float(), tokens


@torch.no_grad()
def build_bank(model, data_root, eval_tf, batch_size, workers, device, use_bf16):
    ds = ImageListDataset(data_root, "train", transform=eval_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    feats, labels = [], []
    for images, y, _ in tqdm(loader, desc="bank (training photos)"):
        f, _, _ = features_and_logits(model, images.to(device, non_blocking=True), use_bf16)
        feats.append(f.half().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def class_index(labels, num_classes):
    """(C, L) row indices of each class's bank photos, padded with -1; L = most photos in a class."""
    rows = defaultdict(list)
    for i, c in enumerate(labels.tolist()):
        rows[c].append(i)
    width = max(len(v) for v in rows.values())
    idx = torch.full((num_classes, width), -1, dtype=torch.long)
    for c, v in rows.items():
        idx[c, :len(v)] = torch.tensor(v)
    return idx


def knn_scores(f, cand, bank, idx, m):
    """f (B, D) photo features, cand (B, K) classes -> (B, K) mean of the m best cosines."""
    rows = idx[cand]                                      # (B, K, L)
    valid = rows >= 0
    sims = torch.einsum("bd,bkld->bkl", f, bank[rows.clamp_min(0)].float())
    sims = sims.masked_fill(~valid, -2.0)
    best = sims.topk(min(m, sims.shape[-1]), dim=-1).values
    return best.masked_fill(best <= -2.0, float("nan")).nanmean(-1).nan_to_num(-1.0)


def rerank(lp_full, f, bank, idx, k, m, beta):
    lp, cand = lp_full.topk(k, dim=1)
    score = lp + beta * knn_scores(f, cand, bank, idx, m)
    return cand.gather(1, score.argmax(1, keepdim=True)).squeeze(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="run trained on the photos in data_root/train.txt")
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--team", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--bank", default=None, help="cache file for the training-photo features (built if missing)")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--betas", type=float, nargs="+", default=[20.0],
                   help="kNN weights; the first is the main submission")
    p.add_argument("--zoom", action="store_true", help="re-rank Van's zoom probabilities (src/van_zoom.py)")
    p.add_argument("--crop_pct", type=float, default=None, help="eval crop for the photo (Van's 448 zoom: 0.9)")
    p.add_argument("--zoom_mass", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    apply_overrides(cfg, data_root=args.data_root, batch_size=args.batch_size, num_workers=args.workers)
    device = get_device()
    set_seed(int(cfg.seed))  # as src.train does: same frozen weights if the backbone is not pretrained
    model = build_model(cfg).to(device)
    ckpt = run_dir / "last.pt"
    load_checkpoint(model, ckpt, device)
    model.eval()
    _, bank_tf = build_transforms(cfg)
    use_bf16 = bool(cfg.bf16) and device.type == "cuda"
    bs, nw = int(cfg.batch_size), int(cfg.num_workers)

    bank_path = Path(args.bank) if args.bank else None
    if bank_path and bank_path.is_file():
        blob = torch.load(bank_path, map_location="cpu", weights_only=False)
        if blob.get("checkpoint") != str(ckpt):
            raise SystemExit(f"{bank_path} was built from {blob.get('checkpoint')}, not {ckpt}: use another --bank file")
        bank, bank_y = blob["features"], blob["labels"]
        print(f"[rerank] bank loaded: {len(bank_y):,} training photos from {bank_path}")
    else:
        bank, bank_y = build_bank(model, args.data_root, bank_tf, bs, nw, device, use_bf16)
        if bank_path:
            bank_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"features": bank, "labels": bank_y, "checkpoint": str(ckpt)}, bank_path)
            print(f"[rerank] bank saved: {bank_path}")
    idx = class_index(bank_y, int(cfg.num_classes)).to(device)
    bank = bank.to(device)

    if args.zoom:
        from src import van_zoom

        photo_tf = van_zoom.two_view_transform(cfg, args.crop_pct)
        mass = van_zoom.DEFAULT_MASS if args.zoom_mass is None else float(args.zoom_mass)
    else:
        _, photo_tf = build_transforms(cfg, eval_crop_pct=args.crop_pct)
    ds = ImageListDataset(args.data_root, args.split, transform=photo_tf)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=device.type == "cuda")

    name = f"{args.split}_team_{args.team}"
    variants = ["plain"] + [f"knn_b{b:g}" for b in args.betas]
    if args.zoom:
        variants += ["zoom"] + [f"zoom_knn_b{b:g}" for b in args.betas]
    main_variant = (f"zoom_knn_b{args.betas[0]:g}" if args.zoom else f"knn_b{args.betas[0]:g}")
    preds, paths, labels = defaultdict(list), [], []
    rerank_s = 0.0
    with torch.no_grad():
        for batch, y, rel in tqdm(loader, desc=f"predict {args.split}"):
            small = (batch[0] if args.zoom else batch).to(device, non_blocking=True)
            f, logits, tokens = features_and_logits(model, small, use_bf16)
            lp_plain = logits.log_softmax(1)
            if args.zoom:
                big = batch[1].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    zoomed = van_zoom.zoom_view(model, tokens, logits, big, mass=mass)
                _, logits_z, _ = features_and_logits(model, zoomed, use_bf16)
                probs = logits.softmax(1) + logits_z.softmax(1)        # as src.predict --zoom
                lp_zoom = (probs / 2).clamp_min(1e-12).log()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds["plain"].append(lp_plain.argmax(1).cpu())
            for b in args.betas:
                preds[f"knn_b{b:g}"].append(rerank(lp_plain, f, bank, idx, args.K, args.m, b).cpu())
            if args.zoom:
                preds["zoom"].append(lp_zoom.argmax(1).cpu())
                for b in args.betas:
                    preds[f"zoom_knn_b{b:g}"].append(rerank(lp_zoom, f, bank, idx, args.K, args.m, b).cpu())
            if device.type == "cuda":
                torch.cuda.synchronize()
            rerank_s += time.perf_counter() - t0
            paths += list(rel)
            labels.append(y)
    preds = {k: torch.cat(v) for k, v in preds.items()}
    labels = torch.cat(labels)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for v in variants:
        with (out / f"{name}_{v}.txt").open("w", encoding="utf-8") as fh:
            fh.writelines(f"{r} {c}\n" for r, c in zip(paths, preds[v].tolist()))
    with (out / f"{name}.txt").open("w", encoding="utf-8") as fh:
        fh.writelines(f"{r} {c}\n" for r, c in zip(paths, preds[main_variant].tolist()))

    n = len(paths)
    base = "zoom" if args.zoom else "plain"
    stats = {"run_dir": str(run_dir), "split": args.split, "photos": n, "K": args.K, "m": args.m,
             "betas": args.betas, "zoom": args.zoom, "crop_pct": args.crop_pct, "bank_photos": int(len(bank_y)),
             "main_submission": f"{name}.txt = {main_variant}",
             "changed_vs_" + base: {v: int((preds[v] != preds[base]).sum()) for v in variants if v != base},
             "rerank_ms_per_photo": round(1000 * rerank_s / max(n, 1) / max(len(variants) - 1, 1), 4),
             "note": f"rerank time per variant, excludes the forward passes; batch size {bs}"}
    if bool((labels >= 0).all()):  # a labelled split (pilot holdout): score every variant honestly
        acc = {v: round(float((preds[v] == labels).float().mean()), 4) for v in variants}
        classes = sorted(set(labels.tolist()))
        half = torch.tensor([{c: i % 2 for i, c in enumerate(classes)}[c] for c in labels.tolist()])
        tuned = {}
        for mode in (["plain"] + (["zoom"] if args.zoom else [])):
            prefix = "knn_b" if mode == "plain" else "zoom_knn_b"
            gains = []
            for tune, test in ((0, 1), (1, 0)):
                pick = max(args.betas, key=lambda b: float((preds[f"{prefix}{b:g}"] == labels)[half == tune].float().mean()))
                sel = half == test
                after = float((preds[f"{prefix}{pick:g}"] == labels)[sel].float().mean())
                before = float((preds[mode] == labels)[sel].float().mean())
                gains.append({"beta_chosen_on_half": tune, "beta": pick, "test_half_before": round(before, 4),
                              "test_half_after": round(after, 4)})
            tuned[mode] = {"held_out": gains,
                           "held_out_gain_points": round(100 * sum(g["test_half_after"] - g["test_half_before"]
                                                                   for g in gains) / 2, 2)}
        stats["accuracy"] = acc
        stats["knn_gain_with_beta_chosen_on_other_half"] = tuned
    (out / "rerank_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"[rerank] wrote {name}.txt ({main_variant}) and {len(variants)} variant files in {out}")


if __name__ == "__main__":
    main()
