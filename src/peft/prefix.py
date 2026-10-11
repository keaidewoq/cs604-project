"""Direct, independent K/V prefixes for timm's standard ViT attention."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PrefixAttention(nn.Module):
    """Keep original queries; prepend learned keys/values before attention."""

    def __init__(self, attention, length, init_std=0.02):
        super().__init__()
        if length <= 0:
            raise ValueError("prefix_length must be positive")
        if not isinstance(attention.qkv, nn.Linear):
            raise TypeError("Prefix tuning requires fused linear QKV attention")
        if getattr(attention, "gate", None) is not None:
            raise TypeError("Gated attention is not supported by prefix tuning")
        self.base = attention
        self.base.requires_grad_(False)
        self.num_heads = attention.num_heads
        self.head_dim = attention.qkv.out_features // (3 * self.num_heads)
        shape = (self.num_heads, length, self.head_dim)
        self.prefix_key = nn.Parameter(attention.qkv.weight.new_empty(shape))
        self.prefix_value = nn.Parameter(attention.qkv.weight.new_empty(shape))
        nn.init.normal_(self.prefix_key, std=init_std)
        nn.init.normal_(self.prefix_value, std=init_std)
        self.enabled = True

    def forward(self, x, attn_mask=None, is_causal=False):
        if not self.enabled:
            return self.base(x) if attn_mask is None and not is_causal else self.base(
                x, attn_mask=attn_mask, is_causal=is_causal
            )
        if attn_mask is not None or is_causal:
            raise ValueError("PrefixAttention supports unmasked, noncausal ViT attention only")
        batch, tokens, _ = x.shape
        qkv = self.base.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        pk = self.prefix_key.to(dtype=k.dtype).unsqueeze(0).expand(batch, -1, -1, -1)
        pv = self.prefix_value.to(dtype=v.dtype).unsqueeze(0).expand(batch, -1, -1, -1)
        k = torch.cat((pk, k), dim=2)
        v = torch.cat((pv, v), dim=2)
        q = self.base.q_norm(q)
        k = self.base.k_norm(k)
        if self.base.fused_attn:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.base.attn_drop.p if self.training else 0.0,
                scale=self.base.scale,
            )
        else:
            scores = (q * self.base.scale) @ k.transpose(-2, -1)
            out = self.base.attn_drop(scores.softmax(dim=-1)) @ v
        out = out.transpose(1, 2).reshape(batch, tokens, self.num_heads * self.head_dim)
        out = getattr(self.base, "norm", nn.Identity())(out)
        return self.base.proj_drop(self.base.proj(out))


def apply_prefix(model, cfg):
    length = int(cfg.get("prefix_length", 20))
    placement = cfg.get("prefix_placement", "all")
    depth = len(model.blocks)
    counts = {"last1": 1, "last4": 4, "all": depth}
    if placement not in counts or counts[placement] > depth:
        raise ValueError("prefix_placement must be last1, last4, or all (within model depth)")
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    original_backbone = sum(p.numel() for p in model.parameters() if id(p) not in head_ids)
    # Prefix initialization must not change the trainer's data RNG stream.
    with torch.random.fork_rng(devices=[]):
        for block in model.blocks[depth - counts[placement]:]:
            block.attn = PrefixAttention(block.attn, length, float(cfg.get("prefix_init_std", 0.02)))
    adaptation = sum(
        p.numel() for p in model.parameters() if p.requires_grad and id(p) not in head_ids
    )
    pct = 100 * adaptation / original_backbone
    cap = float(cfg.get("max_adaptation_pct", 5.0))
    if pct >= cap:
        raise ValueError(f"Prefix parameters {pct:.6f}% must be strictly below {cap}%")
    model.adaptation_stats = {
        "adaptation_parameters": adaptation,
        "original_backbone_parameters": original_backbone,
        "adaptation_pct": pct,
    }
    print(f"[prefix] placement={placement}, length={length}, backbone trainable={pct:.6f}%")
    return model
