"""Build ViT-B/16 variants used by the CS604 experiments."""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A frozen linear layer plus a small trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return self.base(inputs) + update * self.scaling


class CosineClassifier(nn.Module):
    """Cosine-similarity head suitable for class-prototype initialization."""

    def __init__(self, in_features: int, out_features: int, scale: float = 20.0, force_fp32: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.force_fp32 = force_fp32
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.force_fp32:
            # Match build_prototypes.py: keep backbone AMP, but normalize and
            # compare features outside autocast so nearby class scores don't tie.
            with torch.autocast(device_type=inputs.device.type, enabled=False):
                scale = self.log_scale.float().exp().clamp(max=100.0)
                return scale * F.linear(
                    F.normalize(inputs.float(), dim=-1),
                    F.normalize(self.weight.float(), dim=-1),
                )
        scale = self.log_scale.exp().clamp(max=100.0)
        return scale * F.linear(F.normalize(inputs, dim=-1), F.normalize(self.weight, dim=-1))


def _classifier(model: nn.Module) -> nn.Module:
    classifier = model.get_classifier()
    if not isinstance(classifier, nn.Module):
        raise TypeError("The timm model does not expose a classifier module")
    return classifier


def _replace_qkv_with_lora(
    model: nn.Module, rank: int, alpha: float, dropout: float
) -> int:
    replaced = 0
    for block in model.blocks:
        qkv = block.attn.qkv
        if not isinstance(qkv, nn.Linear):
            raise TypeError(f"Expected Linear qkv, found {type(qkv).__name__}")
        block.attn.qkv = LoRALinear(qkv, rank=rank, alpha=alpha, dropout=dropout)
        replaced += 1
    return replaced


def build_vit(
    checkpoint: str,
    num_classes: int,
    grad_checkpointing: bool = False,
    mode: str = "full",
    head_type: str = "linear",
    train_head: bool = True,
    last_blocks: int = 1,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    cosine_head_fp32: bool = False,
) -> nn.Module:
    model = timm.create_model(checkpoint, pretrained=True, num_classes=num_classes)
    if head_type == "cosine":
        old_head = _classifier(model)
        if not isinstance(old_head, nn.Linear):
            raise TypeError("Cosine head replacement expects a linear timm classifier")
        model.head = CosineClassifier(old_head.in_features, num_classes, force_fp32=cosine_head_fp32)
    elif head_type != "linear":
        raise ValueError(f"Unknown head_type: {head_type}")

    if grad_checkpointing:
        model.set_grad_checkpointing(True)

    if mode == "full":
        model.requires_grad_(True)
    elif mode == "head":
        model.requires_grad_(False)
        _classifier(model).requires_grad_(True)
    elif mode == "last_blocks":
        if not 1 <= last_blocks <= len(model.blocks):
            raise ValueError(f"last_blocks must be in [1, {len(model.blocks)}]")
        model.requires_grad_(False)
        for block in model.blocks[-last_blocks:]:
            block.requires_grad_(True)
        model.norm.requires_grad_(True)
        _classifier(model).requires_grad_(train_head)
    elif mode == "lora":
        model.requires_grad_(False)
        _replace_qkv_with_lora(model, lora_rank, lora_alpha, lora_dropout)
        _classifier(model).requires_grad_(train_head)
    else:
        raise ValueError(f"Unknown adaptation mode: {mode}")
    return model


def apply_class_prototypes(model: nn.Module, prototypes: torch.Tensor) -> None:
    """Copy one normalized feature average into each class weight."""

    classifier = _classifier(model)
    weight = getattr(classifier, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("The classifier has no weight parameter")
    if tuple(weight.shape) != tuple(prototypes.shape):
        raise ValueError(
            f"Prototype shape {tuple(prototypes.shape)} does not match "
            f"classifier weight {tuple(weight.shape)}"
        )
    normalized = F.normalize(prototypes.float(), dim=1).to(
        device=weight.device, dtype=weight.dtype
    )
    with torch.no_grad():
        weight.copy_(normalized)
        bias = getattr(classifier, "bias", None)
        if isinstance(bias, torch.Tensor):
            bias.zero_()


def build_from_config(cfg: dict) -> nn.Module:
    """Build a model from the shared YAML configuration dictionary."""

    return build_vit(
        checkpoint=cfg["timm_checkpoint"],
        num_classes=int(cfg["num_classes"]),
        grad_checkpointing=bool(cfg.get("grad_checkpointing", False)),
        mode=str(cfg.get("mode", "full")),
        head_type=str(cfg.get("head_type", "linear")),
        train_head=bool(cfg.get("train_head", True)),
        last_blocks=int(cfg.get("last_blocks", 1)),
        lora_rank=int(cfg.get("lora_rank", 8)),
        lora_alpha=float(cfg.get("lora_alpha", 16.0)),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        cosine_head_fp32=bool(cfg.get("cosine_head_fp32", False)),
    )


def parameter_report(model: nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    classifier_ids = {id(p) for p in _classifier(model).parameters()}
    classifier_total = sum(p.numel() for p in model.parameters() if id(p) in classifier_ids)
    classifier_trainable = sum(
        p.numel() for p in model.parameters() if id(p) in classifier_ids and p.requires_grad
    )
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
        "classifier_params": classifier_total,
        "classifier_trainable_params": classifier_trainable,
        "backbone_trainable_params": trainable - classifier_trainable,
    }
