"""Linear probe on cached frozen features (Ana's augreg2 probe).

Embeds every train / val / test image once at `img_size`, then trains only a
linear head on those features: an lr sweep scored on a 5-per-class holdout,
then a retrain on all train images with the best lr. Much faster than
`method: linear_probe` in the trainer, which re-runs the backbone every epoch.

Writes into a run folder under `output_dir`: the sweep + per-epoch curves,
`probe_head.pt` (standardisation folded in, so it drops straight onto the
model as a starting head for fine-tuning, i.e. LP-FT), and `{split}_team_{team}.txt`
plus a CodaBench-ready zip.

  python scripts/linear_probe_cached.py --config configs/linear_probe_cached.yaml \\
      --data_root /content/data --output_dir /content/drive/MyDrive/CS604_RUNS --team 08
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ImageListDataset, build_transforms
from src.model import build_model
from src.utils import (
    Timer,
    VRAMTracker,
    apply_overrides,
    get_device,
    gpu_hours,
    load_config,
    save_config,
    set_seed,
)


def holdout_mask(labels, per_class, seed):
    """Same split as Ana's notebooks: shuffle with `seed`, first `per_class` of each class."""
    df = pd.DataFrame({"label": labels})
    shuffled = df.sample(frac=1, random_state=int(seed))
    idx = shuffled.index[(shuffled.groupby("label").cumcount() < int(per_class)).values]
    mask = np.zeros(len(labels), dtype=bool)
    mask[idx] = True
    return mask


@torch.no_grad()
def extract_features(model, ds, cache_path, cfg, device):
    """Raw (unnormalised) pre-logit features, cached as float16 .npy on local disk."""
    if cache_path.is_file():
        print(f"[probe] using cached {cache_path}")
        return np.load(cache_path)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("feature_batch_size", 256)),
        shuffle=False,
        num_workers=int(cfg.get("feature_workers") or os.cpu_count() or 2),
        pin_memory=device.type == "cuda",
    )
    use_bf16 = bool(cfg.bf16) and device.type == "cuda"
    out = []
    for images, _, _ in tqdm(loader, desc=f"features {ds.split}"):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            feats = model.forward_head(model.forward_features(images), pre_logits=True)
        out.append(feats.float().cpu())
    feats = torch.cat(out).numpy().astype(np.float16)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, feats)
    return feats


class Standardise:
    """Feature standardisation with statistics from the fit images only.

    With `enabled=False` it is the identity (mu=0, sd=1), so the fold below is a no-op.
    """

    def __init__(self, X, enabled=True):
        if enabled:
            self.mu = X.mean(0, keepdim=True)
            self.sd = X.std(0, keepdim=True) + 1e-6
        else:
            self.mu = torch.zeros(1, X.shape[1], device=X.device)
            self.sd = torch.ones(1, X.shape[1], device=X.device)

    def __call__(self, X):
        return (X - self.mu) / self.sd


@torch.no_grad()
def evaluate(head, norm, X, y, smoothing, bs=16384):
    """Holdout loss (same label-smoothed loss as training), top-1 and top-5."""
    loss_sum, c1, c5 = 0.0, 0, 0
    for i in range(0, len(X), bs):
        logits = head(norm(X[i:i + bs]))
        yy = y[i:i + bs]
        loss_sum += F.cross_entropy(logits, yy, label_smoothing=smoothing, reduction="sum").item()
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
        c1 += (top5[:, 0] == yy).sum().item()
        c5 += (top5 == yy[:, None]).any(dim=1).sum().item()
    n = len(X)
    return loss_sum / n, c1 / n, c5 / n


def train_head(X, y, norm, lr, cfg, X_eval=None, y_eval=None):
    """AdamW + OneCycle on a linear head. Logs holdout metrics per epoch if given."""
    set_seed(int(cfg.seed))
    epochs = int(cfg.epochs)
    bs = int(cfg.batch_size)
    smoothing = float(cfg.label_smoothing)
    head = torch.nn.Linear(X.shape[1], int(cfg.num_classes)).to(X.device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=float(cfg.weight_decay))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        total_steps=epochs * math.ceil(len(X) / bs),
        pct_start=min(max(float(cfg.warmup) / epochs, 0.01), 0.99),
    )
    Xn = norm(X)
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(len(Xn), device=Xn.device)
        loss_sum = 0.0
        for i in range(0, len(Xn), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(head(Xn[idx]), y[idx], label_smoothing=smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(idx)
        row = {"epoch": epoch, "lr_now": sched.get_last_lr()[0], "train_loss": loss_sum / len(Xn)}
        if X_eval is not None:
            row["holdout_loss"], row["holdout_top1"], row["holdout_top5"] = evaluate(
                head, norm, X_eval, y_eval, smoothing
            )
        history.append(row)
    return head, pd.DataFrame(history)


def plot_sweep(histories, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for c, (lr, h) in zip(colors, histories.items()):
        axes[0].plot(h.epoch, h.train_loss, color=c, label=f"train, lr={lr:g}")
        axes[0].plot(h.epoch, h.holdout_loss, color=c, linestyle="--", label=f"holdout, lr={lr:g}")
        axes[1].plot(h.epoch, h.holdout_top1, color=c, label=f"lr={lr:g}")
    axes[0].set(title="Loss per epoch (solid = train, dashed = holdout)", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Holdout top-1 per epoch", xlabel="epoch", ylabel="top-1 accuracy")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_submission(run_dir, split, team, paths, preds):
    txt = run_dir / f"{split}_team_{team}.txt"
    with txt.open("w", encoding="utf-8") as f:
        for rel_path, pred in zip(paths, preds):
            f.write(f"{rel_path} {int(pred)}\n")
    with zipfile.ZipFile(txt.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(txt, arcname=txt.name)
    print(f"[probe] wrote {txt} + .zip ({len(paths)} lines)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/linear_probe_cached.yaml")
    parser.add_argument("--team", required=True, help="Team id used in the filename, e.g. 08")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--feature_dir", default=None, help="Local cache for .npy features")
    parser.add_argument("--img_size", type=int, default=None, help="Input resolution, e.g. 224 or 336")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    apply_overrides(
        cfg,
        data_root=args.data_root,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        img_size=args.img_size,
    )
    if str(cfg.get("lr_schedule", "onecycle")) != "onecycle":
        raise ValueError(f"lr_schedule {cfg.lr_schedule!r} not supported; use onecycle")
    # Frozen backbone with a linear head; the probe only needs features.
    apply_overrides(cfg, method="linear_probe", head_type="linear", head_init="random")
    device = get_device()
    if device.type != "cuda":
        print("[probe] warning: no GPU; feature extraction will be very slow")

    run_dir = Path(cfg.output_dir) / f"linear_probe_cached_{cfg.img_size}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    print(f"[probe] run_dir={run_dir}")

    timer = Timer()
    vram = VRAMTracker()
    vram.reset()

    # 1. Frozen features for every split that exists
    _, eval_tf = build_transforms(cfg)
    root = Path(cfg.data_root)
    datasets = {"train": ImageListDataset(root, "train", transform=eval_tf)}
    if not datasets["train"].has_labels:
        raise SystemExit("train.txt must have labels for a linear probe")
    for split in ("val", "test"):
        if (root / f"{split}.txt").exists() or (root / split).is_dir():
            datasets[split] = ImageListDataset(root, split, transform=eval_tf)

    model = build_model(cfg).to(device).eval()
    tag = f"{cfg.model_name}_{cfg.img_size}"
    feat_timer = Timer()
    feats = {
        split: extract_features(model, ds, Path(cfg.feature_dir) / f"{split}_{tag}.npy", cfg, device)
        for split, ds in datasets.items()
    }
    feat_minutes = feat_timer.elapsed() / 60
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[probe] feature dim {feats['train'].shape[1]}, extraction {feat_minutes:.1f} min")

    # 2. lr sweep on the holdout
    labels = np.array([label for _, label in datasets["train"].samples])
    is_holdout = holdout_mask(labels, cfg.holdout_per_class, cfg.seed)
    print(f"[probe] train {len(labels):,} ({(~is_holdout).sum():,} fit, {is_holdout.sum():,} holdout)")

    X = torch.tensor(feats["train"], device=device).float()
    y = torch.tensor(labels, device=device)
    fit = torch.tensor(~is_holdout, device=device)
    ho = torch.tensor(is_holdout, device=device)
    norm = Standardise(X[fit], enabled=bool(cfg.get("standardise_features", True)))

    probe_timer = Timer()
    sweep, histories = [], {}
    for lr in [float(v) for v in cfg.lr_sweep]:
        lr_timer = Timer()
        _, hist = train_head(X[fit], y[fit], norm, lr, cfg, X_eval=X[ho], y_eval=y[ho])
        histories[lr] = hist
        last = hist.iloc[-1]
        sweep.append({
            "lr": lr,
            "top1": round(float(last.holdout_top1), 4),
            "top5": round(float(last.holdout_top5), 4),
            "train_loss": round(float(last.train_loss), 3),
            "holdout_loss": round(float(last.holdout_loss), 3),
            "peak_top1": round(float(hist.holdout_top1.max()), 4),
            "peak_epoch": int(hist.holdout_top1.idxmax()),
            "seconds": round(lr_timer.elapsed()),
        })
        print(f"[probe] {sweep[-1]}")
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(run_dir / "probe_sweep.csv", index=False)
    pd.concat([h.assign(lr=lr) for lr, h in histories.items()]).to_csv(
        run_dir / "probe_curves.csv", index=False
    )
    plot_sweep(histories, run_dir / "probe_curves.png")
    best = sweep.loc[sweep.top1.idxmax()]
    best_lr = float(best.lr)
    print(sweep.to_string(index=False))
    print(f"[probe] best lr={best_lr}  holdout top-1 {best.top1:.4f}")
    if best_lr == max(sweep.lr):
        print("[probe] note: best lr is the largest tried; a higher one might do slightly better")

    # 3. Retrain on every train image with the best lr, then predict
    final_head, _ = train_head(X, y, norm, best_lr, cfg)
    probe_minutes = probe_timer.elapsed() / 60

    for split in ("val", "test"):
        if split not in feats:
            continue
        Xs = torch.tensor(feats[split], device=device).float()
        with torch.no_grad():
            preds = final_head(norm(Xs)).argmax(dim=1).cpu().numpy()
        paths = [p for p, _ in datasets[split].samples]
        write_submission(run_dir, split, args.team, paths, preds)

    # Fold the standardisation into the weights so the head takes raw features:
    # W((x - mu)/sd) + b  =  (W/sd) x + (b - W(mu/sd))
    with torch.no_grad():
        W = final_head.weight / norm.sd
        b = final_head.bias - (final_head.weight * (norm.mu / norm.sd)).sum(dim=1)
    torch.save(
        {"weight": W.cpu(), "bias": b.cpu(), "model_name": cfg.model_name,
         "img_size": int(cfg.img_size), "lr": best_lr},
        run_dir / "probe_head.pt",
    )

    elapsed = timer.elapsed()
    summary = {
        "model_name": cfg.model_name,
        "img_size": int(cfg.img_size),
        "best_lr": best_lr,
        "holdout_top1": float(best.top1),
        "holdout_top5": float(best.top5),
        "feature_minutes": round(feat_minutes, 1),
        "probe_minutes": round(probe_minutes, 1),
        "trainable_params": W.numel() + b.numel(),
        "wall_s": round(elapsed, 1),
        "gpu_hours": round(gpu_hours(elapsed), 4),
        "peak_vram_gb": round(vram.peak_gb(), 3),
    }
    (run_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[probe] {summary}")
    print(f"[probe] saved head: {run_dir / 'probe_head.pt'}")


if __name__ == "__main__":
    main()
