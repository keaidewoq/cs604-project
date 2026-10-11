"""BitFit: selected backbone biases plus the already-trainable classifier."""

from __future__ import annotations


def matches_bias(name, scope):
    if not name.endswith(".bias"):
        return False
    if scope == "all":
        return True
    if scope == "attention":
        return ".attn." in name
    if scope == "mlp":
        return ".mlp." in name
    if scope == "layernorm":
        return name == "norm.bias" or name.endswith((".norm1.bias", ".norm2.bias"))
    return False


def apply_bitfit(model, cfg):
    scope = str(cfg.get("bitfit_scope", "all"))
    if scope not in {"all", "attention", "mlp", "layernorm"}:
        raise ValueError(f"Unknown bitfit_scope: {scope!r}")
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    count = 0
    for name, param in model.named_parameters():
        if id(param) in head_ids:
            continue
        param.requires_grad = matches_bias(name, scope)
        count += int(param.requires_grad)
    print(f"[bitfit] scope={scope}: unfroze {count} backbone bias tensors; classifier also trainable")
    return model
