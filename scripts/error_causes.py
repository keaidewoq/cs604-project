"""What causes the errors? Sort every holdout error of a pilot checkpoint into likely causes.

No training and no hand labels. The checkpoint embeds the pilot's fit photos (its training
photos) and holdout photos; each error is then tested for (causes can overlap):

  sibling       the wrong guess is a close relative: within 5 class IDs (IDs follow the
                taxonomy) or among the true class's 5 most similar classes (prototype cosine)
  unseen_look   the holdout photo is unlike every training photo of its class: its best
                cosine to them is below the 10th percentile of correctly classified photos
                (e.g. a leafy tree when the class's training photos show bare winter trees)
  two_looks     the class's training photos split into two groups (2-means on features;
                split strength in the top 20% of classes, smaller group >= 5 photos) and the
                holdout photo is closest to the SMALLER group (e.g. adult moth when most
                training photos are caterpillars)
  messy_class   at least 20% of the class's training photos sit closer to another class's
                prototype than to their own (leave-one-out): mislabels or off-topic photos

Across all classes it also reports how class consistency, split strength and messiness
relate to per-class holdout accuracy (Spearman), and writes one gallery per cause.

    python scripts/error_causes.py --run_dir <pilot run> --pilot_root /content/pilot_data/c1000_h5_s0 \\
        --out_dir <folder>
writes causes.json, causes_per_error.csv, classes.csv and causes_<cause>.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ImageListDataset, build_transforms
from src.model import build_model
from src.predict import load_checkpoint
from src.utils import apply_overrides, get_device, load_config

CAUSES = ["sibling", "unseen_look", "two_looks", "messy_class"]


@torch.no_grad()
def embed(model, ds, batch_size, workers, device, use_bf16):
    """-> normalised features (N, C), logits argmax top-1 (N,), labels (N,), paths."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    feats, preds, labels, paths = [], [], [], []
    for images, y, rel in tqdm(loader, desc=ds.split):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            f = model.forward_head(model.forward_features(images.to(device, non_blocking=True)), pre_logits=True)
            logits = model.get_classifier()(f)
        feats.append(F.normalize(f.float(), dim=-1).cpu().half())
        preds.append(logits.float().argmax(1).cpu())
        labels.append(y)
        paths += list(rel)
    return torch.cat(feats).float(), torch.cat(preds), torch.cat(labels), paths


def two_means(x, iters=20):
    """2-means on rows of x -> (assignment (n,), share of variance explained by the split)."""
    a = x[0]
    b = x[(x @ a).argmin()]  # farthest from the first point
    centres = torch.stack([a, b])
    for _ in range(iters):
        assign = (x @ centres.T).argmax(1)
        new = torch.stack([x[assign == k].mean(0) if (assign == k).any() else centres[k] for k in range(2)])
        if torch.allclose(new, centres):
            break
        centres = new
    total = ((x - x.mean(0)) ** 2).sum()
    within = sum(((x[assign == k] - x[assign == k].mean(0)) ** 2).sum() for k in range(2) if (assign == k).any())
    return assign, float(1 - within / total.clamp_min(1e-12))


def spearman(a, b):
    ra = torch.tensor(a).argsort().argsort().float()
    rb = torch.tensor(b).argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12))


def gallery(path, items, root, train_by_class, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not items:
        return
    rng = random.Random(0)
    fig, axes = plt.subplots(len(items), 4, figsize=(12, 3.0 * len(items)), squeeze=False)
    for r, it in enumerate(items):
        own = rng.sample(train_by_class[it["true"]], min(2, len(train_by_class[it["true"]])))
        cells = [(it["path"], f"holdout, true {it['true']}, guessed {it['guess']}")]
        cells += [(p, f"training photo of {it['true']}") for p in own]
        guess_photos = train_by_class.get(it["guess"], [])
        cells += [(rng.choice(guess_photos), f"training photo of guess {it['guess']}")] if guess_photos else []
        for c in range(4):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(cells):
                ax.imshow(Image.open(Path(root) / cells[c][0]).convert("RGB").resize((224, 224)))
                ax.set_title(cells[c][1], fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / len(items)))
    fig.savefig(path, dpi=85)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="pilot run (trained on the pilot's fit photos only)")
    p.add_argument("--pilot_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--gallery", type=int, default=8, help="examples per cause")
    args = p.parse_args()

    cfg = load_config(Path(args.run_dir) / "config.yaml")
    apply_overrides(cfg, data_root=args.pilot_root, batch_size=args.batch_size, num_workers=args.workers)
    device = get_device()
    model = build_model(cfg).to(device)
    load_checkpoint(model, Path(args.run_dir) / "last.pt", device)
    model.eval()
    _, eval_tf = build_transforms(cfg)
    use_bf16 = bool(cfg.bf16) and device.type == "cuda"
    bs, nw = int(cfg.batch_size), int(cfg.num_workers)
    fit_f, _, fit_y, fit_paths = embed(model, ImageListDataset(args.pilot_root, "train", eval_tf), bs, nw, device, use_bf16)
    hold_f, hold_pred, hold_y, hold_paths = embed(model, ImageListDataset(args.pilot_root, "val", eval_tf), bs, nw,
                                                  device, use_bf16)

    classes = sorted(set(fit_y.tolist()))
    col = {c: j for j, c in enumerate(classes)}
    idx = defaultdict(list)
    for i, c in enumerate(fit_y.tolist()):
        idx[c].append(i)
    sums = torch.stack([fit_f[idx[c]].sum(0) for c in classes])
    counts = torch.tensor([len(idx[c]) for c in classes]).float()
    protos = F.normalize(sums / counts[:, None], dim=-1)

    # per class: consistency (leave-one-out fit), messiness, two-look split
    stats = {}
    for c in classes:
        x = fit_f[idx[c]]
        loo = F.normalize(sums[col[c]][None] - x, dim=-1)              # own prototype without the photo
        own = (x * loo).sum(1)
        others = x @ protos.T
        others[:, col[c]] = -2
        messy = float((others.max(1).values > own).float().mean())
        assign, split = two_means(x)
        sizes = torch.bincount(assign, minlength=2)
        minority = int(sizes.argmin())
        stats[c] = {"consistency": float(own.mean()), "messy_share": messy, "split": split,
                    "minor_size": int(sizes.min()), "minority": minority, "assign": assign}
    split_cut = torch.tensor([s["split"] for s in stats.values()]).quantile(0.8).item()

    # class neighbours by prototype similarity (for "sibling")
    sim = protos @ protos.T
    sim.fill_diagonal_(-2)
    nearest5 = {c: {classes[j] for j in sim[col[c]].topk(5).indices.tolist()} for c in classes}

    # per holdout photo: best cosine to its own class's training photos
    best_own = torch.tensor([float((fit_f[idx[int(y)]] @ hold_f[i]).max()) for i, y in enumerate(hold_y.tolist())])
    correct = hold_pred == hold_y
    unseen_cut = best_own[correct].quantile(0.10).item()

    rows, right, per_class_acc = [], [], defaultdict(list)
    for i in range(len(hold_y)):
        t, g = int(hold_y[i]), int(hold_pred[i])
        per_class_acc[t].append(int(t == g))
        s = stats[t]
        two = False
        if s["split"] >= split_cut and s["minor_size"] >= 5:
            members = fit_f[idx[t]]
            near = int((members @ hold_f[i]).argmax())
            two = int(s["assign"][near]) == s["minority"]
        flags = {"sibling": t != g and (abs(g - t) <= 5 or g in nearest5[t]),
                 "unseen_look": float(best_own[i]) < unseen_cut,
                 "two_looks": two,
                 "messy_class": s["messy_share"] >= 0.2}
        row = {"path": hold_paths[i], "true": t, "guess": g, "best_cos_to_own_training": round(float(best_own[i]), 4),
               **flags}
        (right if t == g else rows).append(row)

    n = len(rows)
    share = {c: round(sum(r[c] for r in rows) / max(n, 1), 4) for c in CAUSES}
    # the same flags among correctly classified photos: a cause matters if errors have it far more often
    baseline = {c: round(sum(r[c] for r in right) / max(len(right), 1), 4) for c in CAUSES if c != "sibling"}
    none = round(sum(not any(r[c] for c in CAUSES) for r in rows) / max(n, 1), 4)
    pairs = {f"{a}+{b}": round(sum(r[a] and r[b] for r in rows) / max(n, 1), 4)
             for k, a in enumerate(CAUSES) for b in CAUSES[k + 1:]}
    acc = {c: sum(v) / len(v) for c, v in per_class_acc.items()}
    cls = [c for c in classes if c in acc]
    corr = {key: round(spearman([stats[c][key] for c in cls], [acc[c] for c in cls]), 3)
            for key in ("consistency", "messy_share", "split")}
    split_classes = [c for c in cls if stats[c]["split"] >= split_cut and stats[c]["minor_size"] >= 5]
    messy_classes = [c for c in cls if stats[c]["messy_share"] >= 0.2]

    def mean_acc(cs):
        return round(sum(acc[c] for c in cs) / max(len(cs), 1), 4)

    summary = {
        "run_dir": args.run_dir, "holdout_photos": len(hold_y), "errors": n,
        "share_of_errors": share, "same_flag_among_correct_photos": baseline,
        "share_of_errors_with_no_flag": none, "overlaps": pairs,
        "thresholds": {"unseen_look_best_cosine_below": round(unseen_cut, 4), "two_looks_split_at_least": round(split_cut, 4),
                       "messy_class_share_at_least": 0.2},
        "classes": {"total": len(cls), "two_look_classes": len(split_classes), "messy_classes": len(messy_classes),
                    "accuracy_all": mean_acc(cls), "accuracy_two_look_classes": mean_acc(split_classes),
                    "accuracy_messy_classes": mean_acc(messy_classes),
                    "spearman_with_class_accuracy": corr},
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "causes.json").write_text(json.dumps(summary, indent=2))
    with (out / "causes_per_error.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["path"])
        w.writeheader()
        w.writerows(rows)
    with (out / "classes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "holdout_accuracy", "consistency", "messy_share", "split", "smaller_group_size"])
        for c in cls:
            s = stats[c]
            w.writerow([c, round(acc[c], 2), round(s["consistency"], 4), round(s["messy_share"], 3), round(s["split"], 4),
                        s["minor_size"]])
    train_by_class = defaultdict(list)
    for pth, y in zip(fit_paths, fit_y.tolist()):
        train_by_class[y].append(pth)
    for cause in CAUSES:
        picked = [r for r in rows if r[cause]]
        picked = random.Random(0).sample(picked, min(args.gallery, len(picked)))
        gallery(out / f"causes_{cause}.png", picked, args.pilot_root, train_by_class,
                f"{cause}: {share[cause]:.0%} of errors (random examples)")
    print(json.dumps(summary, indent=2))
    print(f"[causes] wrote causes.json, causes_per_error.csv, classes.csv, causes_*.png in {out}")


if __name__ == "__main__":
    main()
