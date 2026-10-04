"""Houlsby-style sequential Adapter (not implemented yet).

Where the bottleneck goes
-------------------------
Insert a residual bottleneck *after* multi-head attention and *after* the FFN
in each Transformer block (two adapters per block, sequential):

    x = x + Adapter_attn(Attention(x))     # after the attention residual
    x = x + Adapter_ffn(FFN(x))            # after the FFN residual

Each Adapter is a down-project (hidden -> adapter_dim) -> activation ->
up-project (adapter_dim -> hidden), with a skip around the bottleneck.
`adapter_dim` comes from the config (see configs/adapter.yaml).

Implementation sketch
---------------------
1. Write an `Adapter` nn.Module (the bottleneck).
2. Wrap or subclass the timm `Block` so both residual streams call it.
3. Freeze the backbone (already done in `apply_method`) and leave adapter
   parameters + the classification head trainable.
"""

from __future__ import annotations


def apply_adapter(model, cfg):
    raise NotImplementedError(
        "Houlsby Adapter is not implemented. Insert residual bottlenecks "
        "after attention and after FFN in each block (see this module's docstring)."
    )
