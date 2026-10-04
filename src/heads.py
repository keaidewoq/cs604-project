"""Classifier heads. Cosine + prototype init come from the van branch."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineClassifier(nn.Module):
    """Logits = scale * cos(feature, class_weight). Good with prototype init."""

    def __init__(self, in_features, out_features, scale=20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x):
        # fp32 compare so nearby classes do not tie under bf16 autocast
        with torch.autocast(device_type=x.device.type, enabled=False):
            scale = self.log_scale.float().exp().clamp(max=100.0)
            return scale * F.linear(
                F.normalize(x.float(), dim=-1),
                F.normalize(self.weight.float(), dim=-1),
            )


def replace_head(model, cfg):
    """Swap timm's linear head for a cosine head when `head_type: cosine`."""
    head_type = str(cfg.get("head_type", "linear"))
    if head_type == "linear":
        return model
    if head_type != "cosine":
        raise ValueError(f"Unknown head_type {head_type!r}")
    old = model.get_classifier()
    if not isinstance(old, nn.Linear):
        raise TypeError(f"Expected nn.Linear classifier, got {type(old)}")
    model.head = CosineClassifier(
        old.in_features, int(cfg.num_classes), scale=float(cfg.get("cosine_scale", 20.0))
    )
    return model


def apply_class_prototypes(model, prototypes):
    """Copy L2-normalized class means into the classifier weight rows."""
    head = model.get_classifier()
    weight = getattr(head, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("Classifier has no weight to fill")
    if tuple(weight.shape) != tuple(prototypes.shape):
        raise ValueError(f"Prototype shape {tuple(prototypes.shape)} != head {tuple(weight.shape)}")
    normalized = F.normalize(prototypes.float(), dim=1).to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        weight.copy_(normalized)
        if getattr(head, "bias", None) is not None:
            head.bias.zero_()


def load_prototypes(cfg, model):
    path = cfg.get("prototype_file")
    if not path or str(cfg.get("head_init", "random")) != "prototype":
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"prototype_file not found: {path}. "
            "Run: python scripts/build_prototypes.py --config configs/prototype_probe.yaml --output <path>"
        )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        built = blob.get("model_name") or blob.get("timm_checkpoint")
        if built and built != cfg.model_name:
            raise ValueError(f"Prototypes were built with {built}, not {cfg.model_name}")
        prototypes = blob["prototypes"] if "prototypes" in blob else blob
    else:
        prototypes = blob
    apply_class_prototypes(model, prototypes)
    print(f"[head] loaded prototypes from {path}  shape={tuple(prototypes.shape)}")
    return prototypes
