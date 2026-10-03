"""AdaptFormer: bottleneck *parallel* to the FFN (not implemented yet).

Where the bottleneck goes
-------------------------
Unlike Houlsby adapters, AdaptFormer does **not** touch the attention path.
A bottleneck sits beside the FFN and is added with a scale `s`:

    y = FFN(x) + s * Adapter(x)

Adapter is down-project (hidden -> adapter_dim) -> ReLU -> up-project
(adapter_dim -> hidden). `adapter_dim` and `adaptformer_scale` come from
configs/adaptformer.yaml.

Implementation sketch
---------------------
1. Write the parallel bottleneck module.
2. Replace or wrap `block.mlp` so the forward adds the scaled adapter output.
3. Freeze the backbone (already done in `apply_method`) and leave adapter
   parameters + the classification head trainable.
"""

from __future__ import annotations


def apply_adaptformer(model, cfg):
    raise NotImplementedError(
        "AdaptFormer is not implemented. Add a scaled bottleneck in parallel "
        "with the FFN in each block (see this module's docstring)."
    )
