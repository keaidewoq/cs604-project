"""CodaBench submission with kNN re-ranking of the model's top-K guesses.

For each photo, the model's top K classes (by log-probability) are re-scored with how close
the photo is to each candidate's own training photos:

    score(c) = log p(c) + beta * knn(c),   knn(c) = mean of the m best cosines between the photo's
                                                    feature and c's training-photo features

and the best-scoring candidate is the prediction. Defaults K = 5, m = 3, beta = 20 were chosen
on held-out halves of the all-class holdout (+1.29 points there; report/progress_log.md,
2026-10-26) and are fixed in advance, not tuned on CodaBench. The bank is every photo in
<data_root>/train.txt (all 45 per class), embedded once by the same checkpoint; it can be
cached (--bank) and reused for the test split.

    python scripts/predict_rerank.py --run_dir <full-scale run> --data_root /content/data \\
        --split val --team 08 --out_dir <folder> --bank <folder>/bank.pt
writes <split>_team_<team>.txt (re-ranked; zip it for CodaBench), <split>_team_<team>_plain.txt
(the model's own top-1, same pass, for comparison) and rerank_stats.json (how many predictions
changed, the extra time per photo for the re-ranking step).
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


@torch.no_grad()
def features_and_logits(model, images, use_bf16):
    with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=use_bf16):
        f = model.forward_head(model.forward_features(images), pre_logits=True)
        logits = model.get_classifier()(f)
    return F.normalize(f.float(), dim=-1), logits.float()


@torch.no_grad()
def build_bank(model, data_root, eval_tf, batch_size, workers, device, use_bf16):
    ds = ImageListDataset(data_root, "train", transform=eval_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    feats, labels = [], []
    for images, y, _ in tqdm(loader, desc="bank (training photos)"):
        f, _ = features_and_logits(model, images.to(device, non_blocking=True), use_bf16)
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="full-scale run (trained on all training photos)")
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--team", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--bank", default=None, help="cache file for the training-photo features (built if missing)")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--beta", type=float, default=20.0)
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
    _, eval_tf = build_transforms(cfg)
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
        bank, bank_y = build_bank(model, args.data_root, eval_tf, bs, nw, device, use_bf16)
        if bank_path:
            bank_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"features": bank, "labels": bank_y, "checkpoint": str(ckpt)}, bank_path)
            print(f"[rerank] bank saved: {bank_path}")
    num_classes = int(cfg.num_classes)
    idx = class_index(bank_y, num_classes).to(device)
    bank = bank.to(device)

    ds = ImageListDataset(args.data_root, args.split, transform=eval_tf)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=device.type == "cuda")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path_rr = out / f"{args.split}_team_{args.team}.txt"
    path_plain = out / f"{args.split}_team_{args.team}_plain.txt"
    changed = n = 0
    rerank_s = 0.0
    with path_rr.open("w", encoding="utf-8") as f_rr, path_plain.open("w", encoding="utf-8") as f_plain:
        for images, _, rel in tqdm(loader, desc=f"predict {args.split}"):
            f, logits = features_and_logits(model, images.to(device, non_blocking=True), use_bf16)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            lp, cand = logits.log_softmax(1).topk(args.K, dim=1)
            score = lp + args.beta * knn_scores(f, cand, bank, idx, args.m)
            new = cand.gather(1, score.argmax(1, keepdim=True)).squeeze(1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            rerank_s += time.perf_counter() - t0
            plain = cand[:, 0]
            changed += int((new != plain).sum())
            n += len(rel)
            for r, a, b in zip(rel, new.tolist(), plain.tolist()):
                f_rr.write(f"{r} {a}\n")
                f_plain.write(f"{r} {b}\n")

    stats = {"run_dir": str(run_dir), "split": args.split, "photos": n, "K": args.K, "m": args.m, "beta": args.beta,
             "bank_photos": int(len(bank_y)), "predictions_changed": changed,
             "share_changed": round(changed / max(n, 1), 4),
             "rerank_ms_per_photo": round(1000 * rerank_s / max(n, 1), 4),
             "note": "rerank time excludes the model's forward pass; at batch size " + str(bs)}
    (out / "rerank_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"[rerank] wrote {path_rr} (re-ranked) and {path_plain} (plain top-1)")


if __name__ == "__main__":
    main()
