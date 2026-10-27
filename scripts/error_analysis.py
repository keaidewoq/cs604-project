"""Where do the errors come from? Score pilot checkpoints on their own holdout, no training.

Use checkpoints that never saw the holdout: pilot runs (trained on the pilot's fit images),
not full-scale runs (trained on all 45 images per class, holdout included).

For each model (--run NAME=RUN_DIR, repeatable) on <pilot_root>/val.txt:
  - top-1 ... top-10 accuracy, and how far down the list the true class sits when wrong
    (the step from top-k to top-(k+1) is the share of photos whose answer was the
    (k+1)-th guess: the room a better choice among the model's candidates could recover);
  - near5: prediction within 5 class IDs of the label (IDs follow the taxonomy, so roughly
    "right genus"), and for errors, whether the guess and the runner-up are relatives;
  - errors_<name>.csv: every misclassified photo with its top-3 guesses, probabilities, the
    true class's rank and an empty `category` column for hand labelling;
  - gallery_<name>.png: --gallery random errors, each row: the photo, its top-3 guesses, a
    training photo of the true class and one of the predicted class;
with two or more models, overlap.json: which errors each model fixes for the other.
Writes topk.png and summary.json too.

    python scripts/error_analysis.py --pilot_root /content/pilot_data/c1000_h5_s0 \\
        --data_root /content/data --out_dir <folder> \\
        --run lora=<CS604_RUNS>/ablation_ablation_train_crops/runs/lora_... \\
        --run full_ft=<CS604_RUNS>/ablation_ablation_param_selection/runs/full_ft_...
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
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ImageListDataset, build_transforms
from src.model import build_model
from src.predict import load_checkpoint
from src.utils import apply_overrides, get_device, load_config, set_seed

TOPK = 10
NEAR = 5


def read_list(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rows.append((parts[0], int(parts[1])))
    return rows


@torch.no_grad()
def score(run_dir, pilot_root, batch_size, workers, crop_pct):
    """-> (paths, labels (N,), top10 classes (N, 10), top10 probs (N, 10), true-class rank (N,))."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    apply_overrides(cfg, data_root=str(pilot_root), batch_size=batch_size, num_workers=workers)
    device = get_device()
    set_seed(int(cfg.seed))  # as src.train does: same frozen weights if the backbone is not pretrained
    model = build_model(cfg).to(device)
    ckpt = run_dir / "last.pt"
    load_checkpoint(model, ckpt, device)
    model.eval()
    _, eval_tf = build_transforms(cfg, eval_crop_pct=crop_pct)
    ds = ImageListDataset(pilot_root, "val", transform=eval_tf)
    if not ds.has_labels:
        raise SystemExit(f"{pilot_root}/val.txt has no labels: use the pilot holdout, not the official val split")
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=int(cfg.num_workers),
                        pin_memory=device.type == "cuda")
    use_bf16 = bool(cfg.bf16) and device.type == "cuda"
    paths, labels, top_i, top_p, ranks = [], [], [], [], []
    for images, y, rel in tqdm(loader, desc=run_dir.name):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(images.to(device, non_blocking=True))
        probs = logits.float().softmax(1).cpu()
        p, i = probs.topk(TOPK, dim=1)
        true_p = probs.gather(1, y[:, None])
        ranks.append((probs > true_p).sum(1) + 1)  # 1 = first guess
        paths += list(rel)
        labels.append(y)
        top_i.append(i)
        top_p.append(p)
    print(f"[errors] {run_dir.name}: {len(paths)} holdout photos, checkpoint {ckpt}")
    return paths, torch.cat(labels), torch.cat(top_i), torch.cat(top_p), torch.cat(ranks)


def summarise(labels, top_i, ranks, pilot_classes):
    n = len(labels)
    wrong = top_i[:, 0] != labels
    curve = {f"top{k}": round(float((ranks <= k).float().mean()), 4) for k in range(1, TOPK + 1)}
    steps = {f"rank{k}": round(float((ranks == k).float().mean()), 4) for k in range(1, TOPK + 1)}
    near = (top_i[:, 0] - labels).abs() <= NEAR
    second_near = (top_i[:, 1] - labels).abs() <= NEAR
    in_pilot = torch.tensor([int(c) in pilot_classes for c in top_i[:, 0].tolist()])
    w = max(int(wrong.sum()), 1)
    return {
        "photos": n,
        "top_k": curve,
        "share_whose_answer_was_guess_k": steps,
        "share_answer_beyond_top10": round(float((ranks > TOPK).float().mean()), 4),
        "median_rank_when_wrong": float(ranks[wrong].float().median()) if wrong.any() else None,
        "near5_accuracy": round(float(near.float().mean()), 4),
        "errors": int(wrong.sum()),
        "errors_guess_is_relative_within_5_ids": round(float((near & wrong).sum()) / w, 4),
        "errors_answer_was_second_guess": round(float(((ranks == 2) & wrong).sum()) / w, 4),
        "errors_answer_in_top5": round(float(((ranks <= 5) & wrong).sum()) / w, 4),
        "correct_runner_up_is_relative": round(float((second_near & ~wrong).sum()) / max(int((~wrong).sum()), 1), 4),
        "errors_guess_outside_pilot_classes": round(float((~in_pilot & wrong).sum()) / w, 4),
    }


def gallery(path, rows, data_root, examples, size, cols_title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.2 * len(rows)), squeeze=False)
    rng = random.Random(0)
    for r, row in enumerate(rows):
        guesses = "\n".join(f"{c}: {p:.0%}" + ("  (true)" if c == row["true"] else "")
                            for c, p in zip(row["top3"], row["top3_p"]))
        cells = [(row["path"], f"holdout photo, true {row['true']}\n(true class rank {row['rank']})"),
                 (None, f"top-3 guesses\n{guesses}"),
                 (rng.choice(examples.get(row["true"], [None])), f"training photo of true {row['true']}"),
                 (rng.choice(examples.get(row["top3"][0], [None])), f"training photo of guess {row['top3'][0]}")]
        for c, (img, title) in enumerate(cells):
            ax = axes[r, c]
            ax.axis("off")
            if img is not None:
                ax.imshow(Image.open(Path(data_root) / img).convert("RGB").resize((size, size)))
                ax.set_title(title, fontsize=8)
            else:
                ax.text(0.05, 0.5, title, fontsize=10, va="center", family="monospace")
    fig.suptitle(cols_title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / len(rows)))
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot_root", required=True, help="pilot data root with a labelled val.txt (the holdout)")
    p.add_argument("--data_root", required=True, help="full data root, for training photos of any class")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR")
    p.add_argument("--crop_pct", type=float, default=None, help="eval centre crop (default: the model's, 0.9)")
    p.add_argument("--gallery", type=int, default=40, help="random errors shown per model")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pilot_classes = {c for _, c in read_list(Path(args.pilot_root) / "train.txt")}
    examples = defaultdict(list)
    for path, c in read_list(Path(args.data_root) / "train.txt"):
        examples[c].append(path)

    results, wrong_sets, summary = {}, {}, {}
    for spec in args.run:
        name, _, run_dir = spec.partition("=")
        paths, labels, top_i, top_p, ranks = score(run_dir, args.pilot_root, args.batch_size, args.workers,
                                                   args.crop_pct)
        results[name] = (paths, labels, top_i, top_p, ranks)
        summary[name] = {"run_dir": run_dir, **summarise(labels, top_i, ranks, pilot_classes)}
        rows = []
        for k in range(len(paths)):
            if int(top_i[k, 0]) == int(labels[k]):
                continue
            rows.append({"path": paths[k], "true": int(labels[k]), "rank": int(ranks[k]),
                         "top3": [int(c) for c in top_i[k, :3]], "top3_p": [round(float(v), 3) for v in top_p[k, :3]],
                         "guess_is_relative": abs(int(top_i[k, 0]) - int(labels[k])) <= NEAR})
        wrong_sets[name] = {r["path"] for r in rows}
        with (out / f"errors_{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "true", "true_rank", "guess1", "p1", "guess2", "p2", "guess3", "p3",
                        "guess_is_relative", "category"])
            for r in rows:
                w.writerow([r["path"], r["true"], r["rank"], r["top3"][0], r["top3_p"][0], r["top3"][1], r["top3_p"][1],
                            r["top3"][2], r["top3_p"][2], int(r["guess_is_relative"]), ""])
        shown = random.Random(0).sample(rows, min(args.gallery, len(rows)))
        for start in range(0, len(shown), 10):
            gallery(out / f"gallery_{name}_{start // 10 + 1}.png", shown[start:start + 10], args.data_root, examples,
                    224, f"{name}: random misclassified holdout photos ({start + 1}-{start + len(shown[start:start + 10])})")
        print(json.dumps({name: summary[name]}, indent=2))

    names = list(results)
    if len(names) >= 2:
        overlap = {}
        for a in names:
            for b in names:
                if a == b:
                    continue
                fixed = wrong_sets[a] - wrong_sets[b]
                ranks_a = dict(zip(results[a][0], results[a][4].tolist()))
                fixed_ranks = [ranks_a[p_] for p_ in fixed]
                overlap[f"{a}_errors_that_{b}_gets_right"] = {
                    "count": len(fixed), "share_of_" + a + "_errors": round(len(fixed) / max(len(wrong_sets[a]), 1), 4),
                    "where_" + a + "_ranked_the_answer": {"2": sum(r == 2 for r in fixed_ranks),
                                                          "3-5": sum(3 <= r <= 5 for r in fixed_ranks),
                                                          "6+": sum(r > 5 for r in fixed_ranks)}}
        overlap["errors_shared_by_all"] = len(set.intersection(*wrong_sets.values()))
        (out / "overlap.json").write_text(json.dumps(overlap, indent=2))
        summary["overlap"] = overlap
        print(json.dumps(overlap, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ks = list(range(1, TOPK + 1))
    for name in names:
        s = summary[name]
        axes[0].plot(ks, [s["top_k"][f"top{k}"] for k in ks], marker="o", label=name)
        axes[1].bar([k + 0.2 * names.index(name) for k in ks[1:]],
                    [s["share_whose_answer_was_guess_k"][f"rank{k}"] for k in ks[1:]], width=0.2, label=name)
    axes[0].set(xlabel="k", ylabel="top-k accuracy", title="Top-k accuracy (pilot holdout)")
    axes[1].set(xlabel="rank of the true class", ylabel="share of holdout photos",
                title="When wrong: where the answer was")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "topk.png", dpi=110)
    print(f"[errors] wrote summary.json, topk.png, errors_*.csv, gallery_*.png in {out}")


if __name__ == "__main__":
    main()
