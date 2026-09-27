"""Build the course ViT-B/16 and replace the classifier for 10,000 classes."""

from __future__ import annotations

import timm
import torch.nn as nn


def build_vit(checkpoint: str, num_classes: int, grad_checkpointing: bool = False) -> nn.Module:
    model = timm.create_model(checkpoint, pretrained=True, num_classes=num_classes)
    if grad_checkpointing:
        model.set_grad_checkpointing(True)
    return model


def parameter_report(model: nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
    }
