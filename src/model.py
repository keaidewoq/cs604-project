"""Build the timm ViT and apply the chosen fine-tuning method."""

from __future__ import annotations

import timm
import torch.nn as nn

from src.heads import replace_head
from src.peft import build_peft
from src.utils import PEFT_METHODS, format_params

EXTRA_PARTS = ("norm", "bias", "pos_embed", "cls_token")


def _unfreeze_head(model):
    head = model.get_classifier()
    if head is None:
        return
    for param in head.parameters():
        param.requires_grad = True


def _unfreeze_extra(model, parts):
    """Also train small backbone pieces next to a PEFT method (`unfreeze_extra`).

    norm: LayerNorm weights and biases; bias: biases of every Linear / conv
    (BitFit-style); pos_embed, cls_token: the learned embeddings. These are
    flagged `_no_weight_decay`, so AdamW does not pull pretrained values to 0.
    """
    unknown = [p for p in parts if p not in EXTRA_PARTS]
    if unknown:
        raise ValueError(f"Unknown unfreeze_extra {unknown}; choose from {list(EXTRA_PARTS)}")
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    picked = []
    for module in model.modules():
        if "norm" in parts and isinstance(module, nn.LayerNorm):
            picked.extend(p for p in module.parameters(recurse=False))
        if "bias" in parts and isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
            picked.append(module.bias)
    for name in ("pos_embed", "cls_token"):
        if name in parts and isinstance(getattr(model, name, None), nn.Parameter):
            picked.append(getattr(model, name))
    n = 0
    for param in picked:
        if id(param) in head_ids:
            continue
        param.requires_grad = True
        param._no_weight_decay = True
        n += param.numel()
    print(f"[unfreeze_extra] {', '.join(parts)}: {n:,} params")


def _backbone_params(model):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    return [p for p in model.parameters() if id(p) not in head_ids]


def log_backbone_trainable(model, base_total):
    """Trainable backbone params as a share of the original backbone, head excluded.

    `base_total` is counted before PEFT modules are added, so the denominator is
    the pretrained model itself (how the brief reports e.g. LoRA = 0.34%).
    """
    total = base_total
    trainable = sum(p.numel() for p in _backbone_params(model) if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    print(f"[params] backbone trainable {format_params(trainable)}/{format_params(total)} ({pct:.2f}%, head excluded)")
    return trainable, total, pct


def apply_method(model, cfg):
    """Freeze everything except the head, then dispatch on `cfg.method`."""
    for param in model.parameters():
        param.requires_grad = False
    _unfreeze_head(model)

    method = cfg.method
    if method == "full_ft":
        for param in model.parameters():
            param.requires_grad = True
        return model
    if method == "linear_probe":
        return model
    if method in PEFT_METHODS:
        base_total = sum(p.numel() for p in _backbone_params(model))
        model = build_peft(model, cfg)
        if cfg.get("unfreeze_extra"):
            _unfreeze_extra(model, list(cfg.unfreeze_extra))
        log_backbone_trainable(model, base_total)
        return model
    raise ValueError(
        f"Unknown method {method!r}. Expected full_ft, linear_probe, or a PEFT method."
    )


def build_model(cfg):
    # Optional stochastic depth; unset keeps timm's default (none).
    extra = {}
    if cfg.get("drop_path_rate"):
        extra["drop_path_rate"] = float(cfg.drop_path_rate)
    model = timm.create_model(
        cfg.model_name,
        pretrained=bool(cfg.get("pretrained", True)),
        num_classes=int(cfg.num_classes),
        img_size=int(cfg.img_size),
        **extra,
    )
    replace_head(model, cfg)
    return apply_method(model, cfg)
