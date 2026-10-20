"""Van's prediction-time "look, then zoom in" view (opt-in: ``src.predict --zoom``;
also used by ``scripts/predict_zoom.py``).

The image is classified as usual. A class-matched saliency map then picks where the
model found its evidence: the cosine between each final patch token and the
predicted class's head weight. The smallest square window (at least ``min_side``
patches) holding ``mass`` of that map is cut from a 2x-resolution copy of the same
view, resized to the model's input size and classified again; the softmax of the
views is averaged. It helps most when the subject is small and off-centre.

Measured on Van's internal holdout (report/progress_log.md, Van, 2026-10-09), eval
crop 0.75, 2 forward passes each: plain + zoom vs plain + flip 77.95 vs 77.10%
(AdaptFormer at 336) and 79.59 vs 78.97% (full FT at 336); a centred box of the
same size scored 77.67 / 79.43%, so most of the gain is the close-up itself and the
rest is zooming to the right place.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

DEFAULT_MASS = 0.7
DEFAULT_MIN_SIDE = 8
FACTOR = 2


class TwoViews:
    """Returns the eval view and the same field of view at ``FACTOR`` x the resolution."""

    def __init__(self, small, big):
        self.small = small
        self.big = big

    def __call__(self, image):
        return self.small(image), self.big(image)


def two_view_transform(cfg, crop_pct=None):
    from src.data import build_transforms
    from src.utils import Config

    _, small = build_transforms(cfg, eval_crop_pct=crop_pct)
    big_cfg = Config({**cfg._data, "img_size": int(cfg.img_size) * FACTOR})
    _, big = build_transforms(big_cfg, eval_crop_pct=crop_pct)
    return TwoViews(small, big)


def cam_map(model, tokens, predicted):
    prefix = int(getattr(model, "num_prefix_tokens", 1))
    patches = F.normalize(tokens[:, prefix:].float(), dim=-1)
    weights = F.normalize(model.get_classifier().weight[predicted].float(), dim=-1)
    cosine = torch.einsum("bnc,bc->bn", patches, weights)
    return (cosine - cosine.mean(1, keepdim=True)).clamp_min(0)


def select_boxes(saliency, grid, mass, min_side):
    """(batch, grid*grid) map -> (batch, 3) boxes ``top, left, side`` in patches."""
    if not math.isfinite(mass) or not 0 < mass <= 1:
        raise ValueError("zoom_mass must be in (0, 1]")
    if min_side < 1 or saliency.shape[1] != grid * grid:
        raise ValueError("Zoom needs a square patch grid and positive minimum side")
    weights = (saliency / saliency.sum(1, keepdim=True).clamp_min(1e-12)).view(-1, 1, grid, grid)
    batch, device = weights.shape[0], saliency.device
    side = torch.full((batch,), grid, device=device, dtype=torch.long)
    top = torch.zeros(batch, device=device, dtype=torch.long)
    left = torch.zeros(batch, device=device, dtype=torch.long)
    done = torch.zeros(batch, device=device, dtype=torch.bool)
    for size in range(min_side, grid):
        sums = F.avg_pool2d(weights, size, stride=1).flatten(1) * size * size
        best, where = sums.max(1)
        width = grid - size + 1
        take = (best >= mass) & ~done
        side[take] = size
        top[take] = where[take] // width
        left[take] = where[take] % width
        done |= take
    return torch.stack([top, left, side], 1)


def crop_zoom(big, boxes, patch_px, size):
    views = []
    for image, (top, left, side) in zip(big, boxes.tolist()):
        region = image[:, top * patch_px:(top + side) * patch_px, left * patch_px:(left + side) * patch_px]
        views.append(F.interpolate(region[None], size=(size, size), mode="bicubic",
                                   antialias=True, align_corners=False)[0])
    return torch.stack(views)


def zoom_view(model, tokens, logits, big, mass=DEFAULT_MASS, min_side=DEFAULT_MIN_SIDE):
    """Zoomed batch from the plain view's tokens/logits and the high-resolution copy."""
    size = int(model.patch_embed.img_size[0])
    patch = int(model.patch_embed.patch_size[0])
    grid = size // patch
    boxes = select_boxes(cam_map(model, tokens, logits.argmax(1)), grid, mass, min_side)
    return crop_zoom(big, boxes, patch * (big.shape[-1] // size), size)
