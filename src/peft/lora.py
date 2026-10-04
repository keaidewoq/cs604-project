"""Hand-written LoRA on the fused qkv Linear of each ViT block."""

from __future__ import annotations

import math

import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap an existing Linear: y = W x + (x A^T B^T) * (alpha / rank).

    The base weight is frozen. B is zero-initialized so the residual starts at 0.
    """

    def __init__(self, linear, rank, alpha, dropout=0.0):
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

    def forward(self, x):
        base = self.linear(x)
        lora = self.lora_B(self.lora_A(self.dropout(x)))
        return base + lora * self.scaling


def apply_lora(model, cfg):
    rank = int(cfg.get("lora_rank", 8))
    alpha = float(cfg.get("lora_alpha", 16))
    dropout = float(cfg.get("lora_dropout", 0.0))
    if not hasattr(model, "blocks"):
        raise TypeError("LoRA expects a ViT with `model.blocks`")

    n = 0
    for block in model.blocks:
        if not hasattr(block, "attn") or not hasattr(block.attn, "qkv"):
            raise TypeError("LoRA expects each block to have `attn.qkv`")
        block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha, dropout=dropout)
        n += 1
    print(f"[lora] wrapped qkv in {n} blocks (rank={rank}, alpha={alpha}, dropout={dropout})")
    return model
