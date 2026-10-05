"""Serial bottleneck adapters (Houlsby et al., ICML 2019) for timm ViT blocks.

Each block gets two adapters, one after the attention output and one after the
MLP output, applied before the block's residual add:

    x = x + attn(norm1(x))  ->  x = x + A_attn(attn(norm1(x)))
    x = x + mlp(norm2(x))   ->  x = x + A_mlp(mlp(norm2(x)))
    where A(h) = h + up(act(down(h)))

`up` is zero-initialised, so the model starts identical to the pretrained one.
Adapters are attached with forward hooks and registered as `block.adapter_attn` /
`block.adapter_mlp`, so the pretrained parameter names (e.g. blocks.0.attn.qkv.weight)
stay unchanged.

Config keys:
    adapter_dim      bottleneck width (default 64)
    adapter_dropout  dropout inside the bottleneck (default 0.0)
    adapter_blocks   "all" or a list of block indices, e.g. [8, 9, 10, 11]

Params per adapter: 2 * dim * width + width + dim. ViT-B, width 64, all 12 blocks:
24 adapters x 99,136 = 2.38M (~2.8% of the backbone; head excluded).
"""

from __future__ import annotations

import math

import torch.nn as nn


class Bottleneck(nn.Module):
    """down -> activation -> dropout -> up, with up zero-initialised."""

    def __init__(self, dim, width, dropout=0.0):
        super().__init__()
        self.down = nn.Linear(dim, width)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(width, dim)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.drop(self.act(self.down(x))))


def _serial_hook(adapter):
    # Forward hook on a sublayer: replaces its output `out` with out + adapter(out).
    def hook(module, inputs, out):
        return out + adapter(out)

    return hook


def select_blocks(model, cfg, key):
    blocks = list(model.blocks)
    chosen = cfg.get(key, "all")
    if chosen in (None, "all"):
        return list(enumerate(blocks))
    return [(i, blocks[i]) for i in chosen]


def mark_trainable(modules):
    n = 0
    for m in modules:
        for p in m.parameters():
            p.requires_grad = True
            n += p.numel()
    return n


def apply_adapter(model, cfg):
    dim = model.embed_dim
    width = int(cfg.get("adapter_dim", 64))
    dropout = float(cfg.get("adapter_dropout", 0.0))
    added = []
    for _, block in select_blocks(model, cfg, "adapter_blocks"):
        block.adapter_attn = Bottleneck(dim, width, dropout)
        block.adapter_mlp = Bottleneck(dim, width, dropout)
        block.attn.register_forward_hook(_serial_hook(block.adapter_attn))
        block.mlp.register_forward_hook(_serial_hook(block.adapter_mlp))
        added += [block.adapter_attn, block.adapter_mlp]
    n = mark_trainable(added)
    print(f"[adapter] {len(added)} serial adapters, width {width}: {n:,} params")
    return model
