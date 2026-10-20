"""Find the organism in training photos without box labels (scripts/find_boxes.py, scripts/box_sweep.py).

Everything runs on the PRETRAINED backbone; nothing is trained. Each photo is resized whole
to img_size x img_size, so its patch grid (21 x 21 at 336) covers the full photo, and each
patch gets a score: high = organism. The box is the bounding rectangle of the largest
connected blob of positive scores, plus a margin.

A variant chooses how the score map is made:
  score   "ddt": DDT class co-localization (Wei et al., IJCAI 2017 / Pattern Recognition 2019):
                 the first principal component of the patch features of ALL the class's
                 photos, i.e. what the class's photos have in common. Its sign is arbitrary
                 and is set by the boundary prior (photo-edge patches score lower).
          "proto": true-label prototype map: cosine of each patch with the class's mean
                 image feature (the prototype head's row for that class), minus the photo's
                 mean. No sign problem; final-layer tokens only.
  layer   None = final tokens (after the last norm); k = the output of block k (0-based).
          Earlier layers keep more spatial detail.
  mask    drop high-norm outlier patch tokens (norm > `outlier_ratio` x the photo's median):
          supervised and self-supervised ViTs put a few such "artifact" tokens in
          low-information background patches (Darcet et al., ICLR 2024), and they can
          dominate a PCA. Masked patches score 0.
  smooth  3 x 3 average of the score map before thresholding, so single hot patches do not
          form blobs.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

BASELINE = {"name": "ddt_final", "score": "ddt", "layer": None, "mask": False, "smooth": False}


def variant_name(v):
    parts = [v["score"], "final" if v["layer"] is None else f"block{v['layer']}"]
    parts += ["masked"] if v["mask"] else []
    parts += ["smooth"] if v["smooth"] else []
    return "_".join(parts)


def read_train_rows(data_root):
    rows = []
    for line in (Path(data_root) / "train.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rows.append((parts[0], int(parts[1])))
    return sorted(rows, key=lambda r: (r[1], r[0]))  # class by class


class WholeImages(Dataset):
    """Whole photos squashed to size x size (no crop, so the patch grid covers the full photo)."""

    def __init__(self, root, rows, size, mean, std):
        self.root, self.rows = Path(root), rows
        self.tf = T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
                             T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i]
        image = Image.open(self.root / path).convert("RGB")
        return self.tf(image), label, i, torch.tensor(image.size)


class Crops(Dataset):
    """(image index, [x0, y0, x1, y1] fraction) -> that region resized to size x size."""

    def __init__(self, root, rows, items, size, mean, std):
        self.root, self.rows, self.items = Path(root), rows, items
        self.tf = T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
                             T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, box = self.items[k]
        image = Image.open(self.root / self.rows[i][0]).convert("RGB")
        w, h = image.size
        x0, y0, x1, y1 = box
        region = image.crop((int(x0 * w), int(y0 * h), max(int(x0 * w) + 1, round(x1 * w)),
                             max(int(y0 * h) + 1, round(y1 * h))))
        return self.tf(region), self.rows[i][1]


class FeatureTaps:
    """Final patch tokens, image features and chosen blocks' outputs in one forward pass."""

    def __init__(self, model, layers):
        self.model = model
        self.prefix = int(getattr(model, "num_prefix_tokens", 1))
        self.layers = sorted({int(k) for k in layers if k is not None})
        self.out = {}
        self.hooks = [model.blocks[k].register_forward_hook(self._hook(k)) for k in self.layers]

    def _hook(self, k):
        def save(_module, _inputs, output):
            self.out[k] = output
        return save

    def __call__(self, images, use_bf16):
        self.out = {}
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            tokens = self.model.forward_features(images)
            feats = self.model.forward_head(tokens, pre_logits=True)
        result = {"final": tokens[:, self.prefix:].float().cpu(), "cls": feats.float().cpu()}
        for k in self.layers:
            result[k] = self.out[k][:, self.prefix:].float().cpu()
        return result


def largest_blob(mask):
    """Largest 4-connected True region of a (g, g) bool grid -> list of (row, col) or None."""
    g = mask.shape[0]
    seen = torch.zeros_like(mask)
    best = None
    for r in range(g):
        for c in range(g):
            if not mask[r, c] or seen[r, c]:
                continue
            blob, queue = [], deque([(r, c)])
            seen[r, c] = True
            while queue:
                y, x = queue.popleft()
                blob.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < g and 0 <= nx < g and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            if best is None or len(blob) > len(best):
                best = blob
    return best


def border_mask(grid):
    m = torch.zeros(grid, grid, dtype=torch.bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m.flatten()


def outlier_tokens(patches, ratio):
    """(n, P, C) -> (n, P) bool: token norm above `ratio` x that photo's median norm."""
    norms = patches.float().norm(dim=-1)
    return norms > ratio * norms.median(dim=1, keepdim=True).values


def ddt_scores(patches, grid, keep=None):
    """DDT on one class: (n, P, C) patches -> (n, P) scores (positive = shared object).

    `keep` (n, P) bool limits the PCA fit to those tokens (outliers left out)."""
    n, p, c = patches.shape
    x = patches.reshape(-1, c).double()
    fit = x if keep is None else x[keep.reshape(-1)]
    mean = fit.mean(0, keepdim=True)
    fit = fit - mean
    _, vecs = torch.linalg.eigh(fit.T @ fit / max(fit.shape[0], 1))
    proj = ((x - mean) @ vecs[:, -1]).float().view(n, p)
    edge = border_mask(grid)
    if proj[:, edge].mean() > proj[:, ~edge].mean():  # boundary prior picks the sign
        proj = -proj
    return proj


def proto_scores(patches, prototype):
    """True-label prototype map: (n, P, C) final patches, (C,) prototype -> (n, P)."""
    sim = F.normalize(patches.float(), dim=-1) @ F.normalize(prototype.float(), dim=0)
    return sim - sim.mean(1, keepdim=True)


def class_scores(feats, variant, grid, prototype, outlier_ratio):
    """Score maps for one class's photos. `feats` from FeatureTaps, concatenated over the class.

    Returns ((n, P) scores, fraction of tokens masked as outliers)."""
    patches = feats["final"] if variant["layer"] is None else feats[variant["layer"]]
    keep = None
    masked = 0.0
    if variant["mask"]:
        keep = ~outlier_tokens(patches, outlier_ratio)
        masked = float((~keep).float().mean())
    if variant["score"] == "ddt":
        scores = ddt_scores(patches, grid, keep)
    elif variant["score"] == "proto":
        if variant["layer"] is not None:
            raise ValueError("proto scores need the final layer (the prototype lives there)")
        scores = proto_scores(patches, prototype)
    else:
        raise ValueError(f"unknown score {variant['score']!r}")
    if keep is not None:
        scores = scores.masked_fill(~keep, 0.0)
    if variant["smooth"]:
        n = scores.shape[0]
        scores = F.avg_pool2d(scores.view(n, 1, grid, grid), 3, stride=1, padding=1,
                              count_include_pad=False).view(n, -1)
    return scores, masked


def box_from_map(score, grid, margin, min_patches, max_area):
    """(P,) score map -> [x0, y0, x1, y1] fractions of the photo, or None (no useful box)."""
    blob = largest_blob((score > 0).view(grid, grid))
    if blob is None or len(blob) < min_patches:
        return None
    ys, xs = [b[0] for b in blob], [b[1] for b in blob]
    x0, x1 = min(xs) / grid, (max(xs) + 1) / grid
    y0, y1 = min(ys) / grid, (max(ys) + 1) / grid
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    box = [max(0.0, x0 - mx), max(0.0, y0 - my), min(1.0, x1 + mx), min(1.0, y1 + my)]
    if (box[2] - box[0]) * (box[3] - box[1]) > max_area:
        return None  # nearly the whole photo: aiming adds nothing, keep the usual crops
    return box


@torch.no_grad()
def prototype_top1(model, data_root, rows, items, prototypes, size, mean, std, batch_size, workers,
                   device, use_bf16):
    """Top-1 of crops against class prototypes (the zero-shot prototype head), restricted to
    the classes in `prototypes` ({label: (C,) tensor})."""
    from torch.utils.data import DataLoader

    labels = sorted(prototypes)
    mat = F.normalize(torch.stack([prototypes[c] for c in labels]).float(), dim=-1).to(device)
    loader = DataLoader(Crops(data_root, rows, items, size, mean, std), batch_size=batch_size,
                        num_workers=workers)
    correct = total = 0
    for images, targets in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            feats = model.forward_head(model.forward_features(images.to(device)), pre_logits=True)
        pred = (F.normalize(feats.float(), dim=-1) @ mat.T).argmax(1).cpu()
        correct += sum(labels[int(p)] == int(t) for p, t in zip(pred, targets))
        total += len(targets)
    return correct / max(total, 1)
