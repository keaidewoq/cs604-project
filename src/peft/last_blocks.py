"""Selected-layer fine-tuning: unfreeze the last N ViT blocks + final norm.

Listed in the course brief as 'Selected Layer Fine-Tuning'. Ported from van.
"""

from __future__ import annotations


def apply_last_blocks(model, cfg):
    n = int(cfg.get("last_n_blocks", 2))
    if not hasattr(model, "blocks"):
        raise TypeError("last_blocks expects model.blocks")
    total = len(model.blocks)
    if not 1 <= n <= total:
        raise ValueError(f"last_n_blocks must be in [1, {total}], got {n}")
    scope = cfg.get("train_scope", "full")
    if scope not in {"full", "qkv", "attention"}:
        raise ValueError("train_scope must be full, qkv, or attention")
    for block in model.blocks[-n:]:
        if scope == "full":
            modules = [block]
        else:
            modules = [block.attn.qkv]
            if scope == "attention":
                modules.append(block.attn.proj)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
    if scope == "full" and hasattr(model, "norm"):
        for param in model.norm.parameters():
            param.requires_grad = True
    limit = cfg.get("max_backbone_trainable_fraction")
    if limit is not None:
        head_ids = {id(p) for p in model.get_classifier().parameters()}
        backbone = [p for p in model.parameters() if id(p) not in head_ids]
        fraction = sum(p.numel() for p in backbone if p.requires_grad) / sum(p.numel() for p in backbone)
        print(f"[budget] trainable backbone={fraction:.4%}, classifier excluded")
        if fraction >= float(limit):
            raise ValueError(f"Trainable backbone fraction {fraction:.4%} must be < {float(limit):.2%}")
    print(f"[last_blocks] last {n}/{total}, scope={scope}, final norm trained={scope == 'full'} (head already trainable)")
    return model
