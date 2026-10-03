"""Build the timm ViT and apply the chosen fine-tuning method."""

from __future__ import annotations

import timm

from src.peft import build_peft
from src.utils import PEFT_METHODS


def _unfreeze_head(model):
    head = model.get_classifier()
    if head is None:
        return
    for param in head.parameters():
        param.requires_grad = True


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
        return build_peft(model, cfg)
    raise ValueError(
        f"Unknown method {method!r}. Expected full_ft, linear_probe, or a PEFT method."
    )


def build_model(cfg):
    model = timm.create_model(
        cfg.model_name,
        pretrained=bool(cfg.get("pretrained", True)),
        num_classes=int(cfg.num_classes),
        img_size=int(cfg.img_size),
    )
    return apply_method(model, cfg)
