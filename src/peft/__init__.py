"""Parameter-efficient fine-tuning methods.

Add a new method by dropping a file here that exposes `apply_<method>(model, cfg)`
and registering it in `build_peft`. Pair it with a YAML config that sets `method`.
"""

from __future__ import annotations

from src.utils import PEFT_METHODS

# Register here for model dispatch and trainable-only checkpoints.
PEFT_METHODS.add("prefix")


def build_peft(model, cfg):
    method = cfg.method
    if method == "prefix":
        from .prefix import apply_prefix

        return apply_prefix(model, cfg)
    if method == "lora":
        from .lora import apply_lora

        return apply_lora(model, cfg)
    if method == "bitfit":
        from .bitfit import apply_bitfit

        return apply_bitfit(model, cfg)
    if method == "adapter":
        from .adapter import apply_adapter

        return apply_adapter(model, cfg)
    if method == "adaptformer":
        from .adaptformer import apply_adaptformer

        return apply_adaptformer(model, cfg)
    if method == "van_adaptformer":
        from .van_adaptformer import apply_van_adaptformer

        return apply_van_adaptformer(model, cfg)
    if method == "last_blocks":
        from .last_blocks import apply_last_blocks

        return apply_last_blocks(model, cfg)
    raise ValueError(f"Unknown PEFT method: {method!r}")
