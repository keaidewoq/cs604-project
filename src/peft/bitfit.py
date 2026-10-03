"""BitFit: train only bias terms (plus the already-unfrozen classification head)."""

from __future__ import annotations


def apply_bitfit(model, cfg):
    """Unfreeze every parameter whose name contains `bias`.

    The head is already trainable from `apply_method`. Everything else stays
    frozen except biases (Linear / LayerNorm / qkv).
    """
    n = 0
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
            n += 1
    print(f"[bitfit] unfroze {n} bias tensors")
    return model
