"""Hand-written LoRA on the Linear layers of each ViT block.

`lora_targets` picks the layers (default: the fused attention qkv only).
Valid names: qkv, proj (attention output), fc1, fc2 (MLP).
"""

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


LORA_TARGETS = {"qkv": "attn", "proj": "attn", "fc1": "mlp", "fc2": "mlp"}


def apply_lora(model, cfg):
    rank = int(cfg.get("lora_rank", 8))
    alpha = float(cfg.get("lora_alpha", 16))
    dropout = float(cfg.get("lora_dropout", 0.0))
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
            wrapped = LoRALinear(getattr(parent, name), rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, name, wrapped)
            params += wrapped.lora_A.weight.numel() + wrapped.lora_B.weight.numel()
            n += 1
    print(
        f"[lora] wrapped {n} layers ({', '.join(targets)} in {len(model.blocks)} blocks), "
        f"rank={rank}, alpha={alpha}, dropout={dropout}: {params:,} params"
    )
    return model
