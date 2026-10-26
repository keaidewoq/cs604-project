"""Van's class-count balancing at prediction time (opt-in: ``src.predict --balance_per_class N``).

The course splits hold the same number of images per class (val 5, test 10; see the
dataset README). Plain argmax ignores that: some classes collect 10+ predictions and
others none. Sinkhorn balancing gives every class a weight ``q_c`` and, over a few
iterations, rescales the weights until each class's expected count (the sum of the
row-normalised, reweighted softmax over all images) is ``N``. Each image is then
assigned the class with the largest reweighted probability. The softmax is kept only
for each image's top ``TOPK`` classes, so memory stays small at 100,000 images.

Measured on Van's internal holdout (exactly 5 images/class, laptop checkpoints at
384, 50 iterations): full FT plain 79.03 -> 80.92%, plain + zoom 79.68 -> 81.35%;
AdaptFormer plain 77.54 -> 79.25%, plain + zoom 78.31 -> 79.82%. A hard assignment
with exactly N per class (min-cost matching) scored lower (80.24 / 80.74 / 78.37 /
78.85%), so the soft version is used.
"""

from __future__ import annotations

import torch

TOPK = 100
DEFAULT_ITERS = 50


def topk(probs: torch.Tensor, k: int = TOPK):
    """(batch, classes) softmax -> top-k probabilities (float32) and class ids (int64), on CPU."""
    p, i = probs.float().topk(k, dim=1)
    return p.cpu(), i.cpu()


def sinkhorn(p: torch.Tensor, i: torch.Tensor, num_classes: int, per_class: float,
             iters: int = DEFAULT_ITERS) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced predictions from top-k ``p`` (N, k) over class ids ``i`` (N, k).

    Returns the predicted class per image and the class weights ``q``.
    """
    p = p.double()
    q = torch.ones(num_classes, dtype=torch.float64)
    flat = i.reshape(-1)
    for _ in range(iters):
        w = p * q[i]
        w = w / w.sum(1, keepdim=True).clamp_min(1e-300)
        counts = torch.zeros(num_classes, dtype=torch.float64).index_add_(0, flat, w.reshape(-1))
        seen = counts > 0  # classes no image has in its top k keep their weight
        q[seen] *= per_class / counts[seen]
    pred = i.gather(1, (p * q[i]).argmax(1, keepdim=True)).squeeze(1)
    return pred, q
