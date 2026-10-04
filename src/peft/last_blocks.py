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
    for block in model.blocks[-n:]:
        for param in block.parameters():
            param.requires_grad = True
    if hasattr(model, "norm"):
        for param in model.norm.parameters():
            param.requires_grad = True
    print(f"[last_blocks] unfroze last {n}/{total} blocks + norm (head already trainable)")
    return model
