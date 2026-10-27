"""AdaptFormer (Chen et al., NeurIPS 2022): a parallel bottleneck beside each MLP.

    x = x + mlp(norm2(x))  ->  x = x + mlp(norm2(x)) + s * up(act(down(norm2(x))))

The adapter shares the MLP's input (the output of norm2), following the paper's
AdaptMLP equation. `up` is zero-initialised, so training starts from the
pretrained model. The adapter is attached with a forward hook and registered as
`block.adaptformer`, so pretrained parameter names stay unchanged.

Config keys:
    adapter_dim         bottleneck width (default 64)
    adaptformer_scale   scale s on the adapter branch (default 0.1, as in the paper)
    adapter_dropout     dropout inside the bottleneck (default 0.0)
    adapter_blocks      "all" or a list of block indices

Params per adapter: 2 * dim * width + width + dim. ViT-B, width 64, all 12 blocks:
12 x 99,136 = 1.19M (~1.4% of the backbone; head excluded).
"""

from __future__ import annotations

from src.peft.adapter import Bottleneck, mark_trainable, select_blocks


def _parallel_hook(adapter, scale):
    # Forward hook on the MLP: inputs[0] is norm2(x), so this adds s * adapter(norm2(x)).
    def hook(module, inputs, out):
        return out + scale * adapter(inputs[0])

    return hook


def apply_adaptformer(model, cfg):
    dim = model.embed_dim
    width = int(cfg.get("adapter_dim", 64))
    scale = float(cfg.get("adaptformer_scale", 0.1))
    dropout = float(cfg.get("adapter_dropout", 0.0))
    added = []
    for _, block in select_blocks(model, cfg, "adapter_blocks"):
        block.adaptformer = Bottleneck(dim, width, dropout)
        block.mlp.register_forward_hook(_parallel_hook(block.adaptformer, scale))
        added.append(block.adaptformer)
    n = mark_trainable(added)
    print(f"[adaptformer] {len(added)} parallel adapters, width {width}, scale {scale}: {n:,} params")
    return model
