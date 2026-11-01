"""Report figures for why the task is hard: one species with very different looks (high
intra-class variation), different species that look the same (inter-class similarity), and the
pair the model confuses most.

Uses a checkpoint's features (the model's own view of the photos), on a shortlist of classes:
  - variation candidates: the --weak weakest classes in class_accuracy.csv;
  - similarity candidates: the --top_pairs top mutual pairs in confused_pairs.csv;
  - the most confused pair: the first pair in confused_pairs.csv confused in both directions.
(class_accuracy.csv and confused_pairs.csv come from scripts/confusion_pairs.py.)

For every shortlisted class, all its training photos are embedded (normalised pre-logit
features). Then:
  - variation: 2-means splits the class into two looks; the class shown is the one whose two
    look centres are least alike (cosine), each look with at least --min_look photos;
  - similarity: a class's prototype is the mean of its features; the pairs shown are the ones
    whose prototypes are most alike;
  - for scale, `showcase.json` compares the two: when a species' two looks are less alike than
    two different species, no single prototype per class can separate them.

Use a checkpoint that did not train on the holdout photos if you also want the red frames to
mean anything (e.g. the all-class holdout LoRA run); the features themselves are fine from any.

Writes intra_class_variation.png, inter_class_similarity.png, most_confused_pair.png and
showcase.json. --classes_only writes just classes.txt (the shortlist), so only those classes
need to be fetched.

    python scripts/showcase_classes.py --run_dir <run> --data_root /content/data \\
        --pairs_csv <dir>/confused_pairs.csv --class_accuracy <dir>/class_accuracy.csv \\
        --errors <dir>/errors_lora_all.csv --out_dir <folder>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.error_causes import two_means
from src.data import build_transforms
from src.model import build_model
from src.predict import load_checkpoint
from src.utils import apply_overrides, get_device, load_config, set_seed


class PhotoList(Dataset):
    def __init__(self, root, paths, transform):
        self.root, self.paths, self.transform = Path(root), paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.root / self.paths[i]).convert("RGB"))


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def shortlist(args):
    """-> (variation candidates, similarity candidate pairs, most confused pair)."""
    weak = [int(r["class"]) for r in read_rows(args.class_accuracy)[:args.weak]]
    rows = read_rows(args.pairs_csv)
    mutual_rows = [r for r in rows if r["mutual"] == "1"]
    mutual = [(int(r["class_a"]), int(r["class_b"])) for r in mutual_rows][:args.top_pairs]
    # Confused both ways, so both classes are in the holdout (a pilot model can guess any class)
    top = mutual_rows[0] if mutual_rows else rows[0]
    return weak, mutual, top


@torch.no_grad()
def embed_classes(args, classes):
    """-> {class: (paths, normalised features (n, C))} for every photo of each class in train.txt."""
    photos = defaultdict(list)
    for line in (Path(args.data_root) / "train.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and int(parts[1]) in classes:
            photos[int(parts[1])].append(parts[0])
    cfg = load_config(Path(args.run_dir) / "config.yaml")
    apply_overrides(cfg, data_root=args.data_root, batch_size=args.batch_size, num_workers=args.workers)
    device = get_device()
    set_seed(int(cfg.seed))  # as src.train does: same frozen weights if the backbone is not pretrained
    model = build_model(cfg).to(device)
    load_checkpoint(model, Path(args.run_dir) / "last.pt", device)
    model.eval()
    _, eval_tf = build_transforms(cfg)
    order = [(c, p) for c in sorted(photos) for p in photos[c]]
    loader = DataLoader(PhotoList(args.data_root, [p for _, p in order], eval_tf),
                        batch_size=int(cfg.batch_size), num_workers=int(cfg.num_workers),
                        pin_memory=device.type == "cuda")
    use_bf16 = bool(cfg.get("bf16", False)) and device.type == "cuda"
    feats = []
    for images in tqdm(loader, desc="embed"):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            f = model.forward_head(model.forward_features(images.to(device, non_blocking=True)), pre_logits=True)
        feats.append(F.normalize(f.float(), dim=-1).cpu())
    feats = torch.cat(feats)
    out, i = {}, 0
    for c in sorted(photos):
        n = len(photos[c])
        out[c] = (photos[c], feats[i:i + n])
        i += n
    return out


def nearest(x, centre, k):
    """Indices of the k rows of x most like `centre`: the most typical photos."""
    return (x @ F.normalize(centre, dim=0)).argsort(descending=True)[:k].tolist()


def draw(path, rows, data_root, per_row, size, title):
    """rows: [(label, [(photo path, frame colour or None, caption or None), ...])]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), per_row, squeeze=False, figsize=(1.9 * per_row, 2.15 * len(rows) + 0.5))
    for r, (label, cells) in enumerate(rows):
        for k in range(per_row):
            ax = axes[r, k]
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if k >= len(cells):
                continue
            p, colour, caption = cells[k]
            img = Image.open(Path(data_root) / p).convert("RGB")
            img.thumbnail((size, size))  # keep the aspect ratio
            ax.imshow(img)
            if colour:
                for side in ax.spines.values():
                    side.set_visible(True)
                    side.set_color(colour)
                    side.set_linewidth(3)
            if caption:
                ax.set_xlabel(caption, fontsize=8, color=colour or "black")
        axes[r, 0].set_ylabel(label, fontsize=9)
    w, h = fig.get_size_inches()
    fig.suptitle(title, fontsize=11, x=0.55 / w, ha="left")
    fig.subplots_adjust(left=0.75 / w, right=1 - 0.05 / w, top=1 - 0.55 / h, bottom=0.3 / h,
                        wspace=0.05, hspace=0.3)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", help="run folder with config.yaml and last.pt")
    p.add_argument("--data_root", help="full data root with train.txt and train/")
    p.add_argument("--pairs_csv", required=True, help="confused_pairs.csv from scripts/confusion_pairs.py")
    p.add_argument("--class_accuracy", required=True, help="class_accuracy.csv from scripts/confusion_pairs.py")
    p.add_argument("--errors", help="errors CSV the pairs came from, for red frames on the most confused pair")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--weak", type=int, default=60, help="weakest classes considered for intra-class variation")
    p.add_argument("--top_pairs", type=int, default=30, help="top mutual pairs considered for similarity")
    p.add_argument("--show_classes", type=int, default=2, help="classes shown for variation")
    p.add_argument("--show_pairs", type=int, default=3, help="pairs shown for similarity")
    p.add_argument("--min_look", type=int, default=8, help="photos the smaller look needs")
    p.add_argument("--per_row", type=int, default=7)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--classes_only", action="store_true", help="only write classes.txt (the shortlist)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weak, mutual, top = shortlist(args)
    top_pair = (int(top["class_a"]), int(top["class_b"]))
    classes = set(weak) | {c for pair in mutual for c in pair} | set(top_pair)
    if args.classes_only:
        (out / "classes.txt").write_text("\n".join(map(str, sorted(classes))) + "\n")
        print(f"[showcase] shortlist of {len(classes)} classes; wrote {out / 'classes.txt'}")
        return
    if not (args.run_dir and args.data_root):
        p.error("--run_dir and --data_root are required unless --classes_only")

    data = embed_classes(args, classes)
    proto = {c: F.normalize(f.mean(0), dim=0) for c, (_, f) in data.items()}
    accuracy = {int(r["class"]): float(r["accuracy"]) for r in read_rows(args.class_accuracy)}

    # Intra-class variation: two looks per class, least alike first
    looks = []
    for c in weak:
        if c not in data:
            continue
        x = data[c][1]
        assign, split = two_means(x)
        sizes = [int((assign == k).sum()) for k in range(2)]
        if min(sizes) < args.min_look:
            continue
        centres = [x[assign == k].mean(0) for k in range(2)]
        looks.append({"class": c, "look_cosine": float(F.cosine_similarity(*centres, dim=0)),
                      "split": round(split, 4), "look_sizes": sizes, "assign": assign, "centres": centres,
                      "consistency": float((x @ proto[c]).mean())})
    looks.sort(key=lambda s: s["look_cosine"])

    # Inter-class similarity: pairs whose prototypes are most alike
    sims = sorted(({"pair": (a, b), "prototype_cosine": float(proto[a] @ proto[b])}
                   for a, b in mutual if a in data and b in data), key=lambda s: -s["prototype_cosine"])
    all_pairs = [float(proto[a] @ proto[b]) for a in proto for b in proto if a < b]

    k = args.per_row
    rows = []
    for s in looks[:args.show_classes]:
        paths, x = data[s["class"]]
        for g in range(2):
            members = (s["assign"] == g).nonzero().flatten()
            pick = [int(members[i]) for i in nearest(x[members], s["centres"][g], k)]
            rows.append((f"class {s['class']}\nlook {g + 1} ({s['look_sizes'][g]} photos)",
                         [(paths[i], None, None) for i in pick]))
    if rows:
        draw(out / "intra_class_variation.png", rows, args.data_root, k, args.size,
             "One species, different looks: each row is one of two groups in the same class "
             "(2-means on the model's features)")

    rows = []
    for s in sims[:args.show_pairs]:
        for c in s["pair"]:
            paths, x = data[c]
            rows.append((f"class {c}\nacc {accuracy.get(c, 1.0):.0%}",
                         [(paths[i], None, None) for i in nearest(x, proto[c], k)]))
    if rows:
        draw(out / "inter_class_similarity.png", rows, args.data_root, k, args.size,
             "Different species, same look: pairs of classes whose prototypes are most alike "
             "(most typical photos of each)")

    wrong = defaultdict(list)
    if args.errors:
        for r in read_rows(args.errors):
            wrong[(int(r["true"]), int(r["guess1"]))].append(r["path"])
    rows = []
    for c, other in (top_pair, top_pair[::-1]):
        paths, x = data[c]
        missed = [q for q in wrong[(c, other)] if q in paths][:k]
        rest = [paths[i] for i in nearest(x, proto[c], k + len(missed)) if paths[i] not in missed]
        cells = [(q, "red", f"called {other}") for q in missed] + [(q, None, None) for q in rest[:k - len(missed)]]
        rows.append((f"class {c}\nacc {accuracy.get(c, 1.0):.0%}", cells))
    draw(out / "most_confused_pair.png", rows, args.data_root, k, args.size,
         f"Most confused pair: {top_pair[0]} called {top_pair[1]} x{top['a_called_b']}, "
         f"{top_pair[1]} called {top_pair[0]} x{top['b_called_a']} (red: holdout photos the model got wrong)")

    summary = {
        "run_dir": args.run_dir,
        "classes_embedded": len(data),
        "intra_class_variation": [{k: v for k, v in s.items() if k not in ("assign", "centres")}
                                  for s in looks[:args.show_classes]],
        "inter_class_similarity": sims[:args.show_pairs],
        "most_confused_pair": {"pair": top_pair, "a_called_b": int(top["a_called_b"]),
                               "b_called_a": int(top["b_called_a"]),
                               "prototype_cosine": float(proto[top_pair[0]] @ proto[top_pair[1]])},
        "for_scale": {
            "median_prototype_cosine_between_shortlisted_classes": round(float(torch.tensor(all_pairs).median()), 4),
            "median_look_cosine_within_weak_classes": round(float(torch.tensor(
                [s["look_cosine"] for s in looks]).median()), 4) if looks else None,
            "median_prototype_cosine_of_mutual_pairs": round(float(torch.tensor(
                [s["prototype_cosine"] for s in sims]).median()), 4) if sims else None,
        },
    }
    (out / "showcase.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[showcase] wrote intra_class_variation.png, inter_class_similarity.png, "
          f"most_confused_pair.png, showcase.json in {out}")


if __name__ == "__main__":
    main()
