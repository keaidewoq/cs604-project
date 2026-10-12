"""Hand-written LoRA on the Linear layers of each ViT block.

`lora_targets` picks the layers (default: the fused attention qkv only).
Valid names: qkv, proj (attention output), fc1, fc2 (MLP).

`lora_dora: true` turns each layer into DoRA (Liu et al., ICML 2024): the adapted
weight W0 + s*B@A is split into a direction (its row-normalised form, updated by
LoRA) and a trainable magnitude per output row, initialised to ||W0|| so training
starts from the pretrained model. As in the paper, the norm is treated as a
constant in the backward pass. Like LoRA, DoRA merges into a plain Linear.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wrap an existing Linear: y = W x + (x A^T B^T) * (alpha / rank).

    The base weight is frozen. B is zero-initialized so the residual starts at 0.
    With dora=True: y = m / ||W0 + s B A|| * (W0 + s B A) x + b, norm per output row.
    """

    def __init__(self, linear, rank, alpha, dropout=0.0, dora=False):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(linear)}")
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        in_f, out_f = linear.in_features, linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.magnitude = None
        if dora:
            self.magnitude = nn.Parameter(linear.weight.detach().float().norm(p=2, dim=1))

    def _adapted_weight(self):
        return self.linear.weight + self.scaling * (self.lora_B.weight @ self.lora_A.weight)

    def forward(self, x):
        lora = self.lora_B(self.lora_A(self.dropout(x)))
        if self.magnitude is None:
            return self.linear(x) + lora * self.scaling
        # DoRA: rescale each output row of the adapted weight to the learned magnitude.
        no_bias = F.linear(x, self.linear.weight)
        norm = self._adapted_weight().float().norm(p=2, dim=1).detach()
        scale = (self.magnitude / norm).to(no_bias.dtype)
        out = scale * (no_bias + lora * self.scaling)
        bias = self.linear.bias
        return out + bias.to(out.dtype) if bias is not None else out

    @torch.no_grad()
    def merged(self):
        """A plain Linear with W + scaling * B @ A folded in (same outputs in eval mode)."""
        base = self.linear
        delta = (self.lora_B.weight.float() @ self.lora_A.weight.float()) * self.scaling
        weight = base.weight.float() + delta
        if self.magnitude is not None:
            weight = weight * (self.magnitude.float() / weight.norm(p=2, dim=1))[:, None]
        out = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None,
                        device=base.weight.device, dtype=base.weight.dtype)
        out.weight.copy_(weight)
        if base.bias is not None:
            out.bias.copy_(base.bias)
        out.requires_grad_(False)
        return out


LORA_TARGETS = {"qkv": "attn", "proj": "attn", "fc1": "mlp", "fc2": "mlp"}


def apply_lora(model, cfg):
    rank = int(cfg.get("lora_rank", 8))
    alpha = float(cfg.get("lora_alpha", 16))
    dropout = float(cfg.get("lora_dropout", 0.0))
    dora = bool(cfg.get("lora_dora", False))
    targets = list(cfg.get("lora_targets", ["qkv"]))
    unknown = [t for t in targets if t not in LORA_TARGETS]
    if unknown:
        raise ValueError(f"Unknown lora_targets {unknown}; choose from {list(LORA_TARGETS)}")
    if not hasattr(model, "blocks"):
        raise TypeError("LoRA expects a ViT with `model.blocks`")

    n = 0
    params = 0
    for block in model.blocks:
        for name in targets:
            parent = getattr(block, LORA_TARGETS[name], None)
            if parent is None or not isinstance(getattr(parent, name, None), nn.Linear):
                raise TypeError(f"LoRA expects each block to have `{LORA_TARGETS[name]}.{name}`")
            wrapped = LoRALinear(getattr(parent, name), rank=rank, alpha=alpha, dropout=dropout, dora=dora)
            setattr(parent, name, wrapped)
            params += sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
            n += 1
    print(
        f"[lora] wrapped {n} layers ({', '.join(targets)} in {len(model.blocks)} blocks), "
        f"rank={rank}, alpha={alpha}, dropout={dropout}{', DoRA' if dora else ''}: {params:,} params"
    )
    return model


def merge_lora(model):
    """Fold every LoRALinear into a plain Linear, for inference only.

    The merged model is a standard ViT, so it runs as fast as the base model.
    LoRA dropout is ignored, which matches eval mode. Do not train after merging.
    """
    n = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merged())
                n += 1
    print(f"[lora] merged {n} LoRA layers into their base weights")
    return model
