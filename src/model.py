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


def _below_block(name, start):
    """True if parameter/module `name` sits before block `start` (embeddings count as before 0)."""
    if start <= 0:
        return False
    if name.startswith("blocks."):
        return int(name.split(".")[1]) < start
    return name.startswith(("patch_embed", "pos_embed", "cls_token", "reg_token", "norm_pre"))


def _unfreeze_extra(model, parts, start=0):
    """Also train small backbone pieces next to a PEFT method (`unfreeze_extra`).

    norm: LayerNorm weights and biases; bias: biases of every Linear / conv
    (BitFit-style); pos_embed, cls_token: the learned embeddings. These are
    flagged `_no_weight_decay`, so AdamW does not pull pretrained values to 0.
    With `start` (first_trainable_block) > 0, pieces before that block stay frozen,
    including pos_embed and cls_token, so backward can stop at block `start`.
    """
    unknown = [p for p in parts if p not in EXTRA_PARTS]
    if unknown:
        raise ValueError(f"Unknown unfreeze_extra {unknown}; choose from {list(EXTRA_PARTS)}")
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    picked = []
    for name, module in model.named_modules():
        if _below_block(name, start):
            continue
        if "norm" in parts and isinstance(module, nn.LayerNorm):
            picked.extend(p for p in module.parameters(recurse=False))
        if "bias" in parts and isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
            picked.append(module.bias)
    for name in ("pos_embed", "cls_token"):
        if name in parts and isinstance(getattr(model, name, None), nn.Parameter) and not _below_block(name, start):
            picked.append(getattr(model, name))
    n = 0
    for param in picked:
        if id(param) in head_ids:
            continue
        param.requires_grad = True
        param._no_weight_decay = True
        n += param.numel()
    kept = f" (blocks {start}+ only; embeddings frozen)" if start > 0 else ""
    print(f"[unfreeze_extra] {', '.join(parts)}: {n:,} params{kept}")


def _check_frozen_below(model, start):
    """With first_trainable_block > 0, nothing before that block may train, or backward runs through it anyway."""
    leaks = [name for name, p in model.named_parameters() if p.requires_grad and _below_block(name, start)]
    if leaks:
        raise ValueError(f"first_trainable_block={start} but these still train: {leaks[:5]}")
    print(f"[frozen] embeddings and blocks 0-{start - 1} frozen; backward stops at block {start}")


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
        start = int(cfg.get("first_trainable_block", 0))
        if cfg.get("unfreeze_extra"):
            _unfreeze_extra(model, list(cfg.unfreeze_extra), start)
        if start > 0:
            _check_frozen_below(model, start)
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
    # Progressive resizing trains some epochs below img_size; timm then resamples
    # pos_embed on the fly (a no-op at img_size, so eval and latency are unchanged).
    if cfg.get("low_res_size"):
        extra["dynamic_img_size"] = True
    model = timm.create_model(
        cfg.model_name,
        pretrained=bool(cfg.get("pretrained", True)),
        num_classes=int(cfg.num_classes),
        img_size=int(cfg.img_size),
        **extra,
    )
    replace_head(model, cfg)
    return apply_method(model, cfg)
