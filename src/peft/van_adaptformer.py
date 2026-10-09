"""Van's AdaptFormer (Chen et al., NeurIPS 2022), as tuned on Van's laptop holdout.

A parallel bottleneck beside every block's MLP:

    x = x + mlp(norm2(x)) + s * up(relu(down(z)))

`z` is the residual stream x (`adapter_input: residual`, the authors' code) or
norm2(x), the MLP's own input (`adapter_input: norm`). `up` starts at zero, so
the model starts as the pretrained one. The adapter is registered as
`block.van_adaptformer` and attached with hooks, so pretrained parameter names
stay unchanged. Separate from `adaptformer.py` (the team's stub) on purpose.

Config keys (method: van_adaptformer):
    adapter_dim              bottleneck width (default 64)
    adapter_scale            scale s (default 0.1, as in the paper)
    adapter_scale_trainable  learn s per block, starting at adapter_scale (default false)
    adapter_dropout          dropout inside the bottleneck (default 0.0)
    adapter_input            residual | norm (default residual)
    adapter_init             lora (Kaiming down, default) | pca (see calibrate_adapters_pca;
                             src/train.py runs it before training)
    adapter_init_batches     training batches for the pca init (default 16)

Params: 12 x (2 x 768 x width + width + 768); width 192 -> 3,550,464 = 4.13% of
ViT-B (head excluded). Building refuses a backbone share of 5% or more.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

ADAPTER_INPUTS = ("residual", "norm")


class VanAdaptFormerAdapter(nn.Module):
    """scale * up(relu(down(x))), with up zero-initialised."""

    def __init__(self, dim, width, scale=0.1, dropout=0.0, scale_trainable=False):
        super().__init__()
        if width <= 0:
            raise ValueError("adapter_dim must be positive")
        self.down = nn.Linear(dim, width)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(width, dim)
        if scale_trainable:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.scale = float(scale)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.dropout(self.act(self.down(x)))) * self.scale


def _attach(block, adapter, adapter_input):
    block.van_adaptformer = adapter
    if adapter_input == "residual":
        def stash(_module, args):
            object.__setattr__(adapter, "_residual", args[0])

        def add(_module, _args, output):
            return output + adapter(adapter.__dict__.pop("_residual"))

        block.norm2.register_forward_pre_hook(stash)
    elif adapter_input == "norm":
        def add(_module, args, output):
            return output + adapter(args[0])
    else:
        raise ValueError(f"adapter_input must be one of {ADAPTER_INPUTS}, got {adapter_input!r}")
    # The output passes through the block's drop_path2 with the MLP's; keep drop_path_rate 0.
    block.mlp.register_forward_hook(add)


def van_adapters(model):
    return [b.van_adaptformer for b in getattr(model, "blocks", ()) if hasattr(b, "van_adaptformer")]


def apply_van_adaptformer(model, cfg):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    backbone_total = sum(p.numel() for p in model.parameters() if id(p) not in head_ids)
    width = int(cfg.get("adapter_dim", 64))
    scale = float(cfg.get("adapter_scale", 0.1))
    adapter_input = str(cfg.get("adapter_input", "residual"))
    added = 0
    for block in model.blocks:
        adapter = VanAdaptFormerAdapter(
            block.mlp.fc1.in_features, width, scale,
            float(cfg.get("adapter_dropout", 0.0)), bool(cfg.get("adapter_scale_trainable", False)),
        )
        _attach(block, adapter, adapter_input)
        for p in adapter.parameters():
            p.requires_grad = True
            added += p.numel()
    trainable_backbone = sum(
        p.numel() for p in model.parameters() if p.requires_grad and id(p) not in head_ids
    )
    pct = 100.0 * trainable_backbone / backbone_total
    print(f"[van_adaptformer] {len(model.blocks)} adapters, width {width}, scale {scale} "
          f"({'learned' if cfg.get('adapter_scale_trainable') else 'fixed'}), input {adapter_input}: "
          f"{added:,} params; backbone trainable {trainable_backbone:,}/{backbone_total:,} "
          f"= {pct:.2f}% (head excluded)")
    if pct >= 5.0:
        raise ValueError(f"van_adaptformer trains {pct:.2f}% of the backbone; the limit is under 5%")
    return model


@torch.no_grad()
def calibrate_adapters_pca(model, batches, device, use_bf16, eps=1e-6):
    """Start each adapter's `down` from the whitened top principal directions of its input.

    Row i of `down` becomes v_i / sqrt(lambda_i), bias -v_i . mean / sqrt(lambda_i): every
    hidden unit starts as a zero-mean, unit-variance projection onto one of the top-`width`
    directions of the tokens that adapter sees. `up` stays zero, so outputs do not change.
    """
    adapters = van_adapters(model)
    if not adapters:
        raise ValueError("adapter_init: pca needs method: van_adaptformer")
    dim, width = adapters[0].down.in_features, adapters[0].down.out_features
    if width > dim:
        raise ValueError(f"adapter_init: pca needs adapter_dim <= {dim}")
    sums = [torch.zeros(dim, dtype=torch.float64, device=device) for _ in adapters]
    outer = [torch.zeros(dim, dim, dtype=torch.float64, device=device) for _ in adapters]
    counts = [0] * len(adapters)

    def collector(index):
        def hook(_module, args):
            tokens = args[0].detach().reshape(-1, dim).double()
            sums[index] += tokens.sum(dim=0)
            outer[index] += tokens.T @ tokens
            counts[index] += tokens.shape[0]
        return hook

    handles = [a.register_forward_pre_hook(collector(i)) for i, a in enumerate(adapters)]
    was_training = model.training
    model.eval()
    enabled = bool(use_bf16) and device.type == "cuda"
    try:
        for images in batches:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
                model(images.to(device, non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if not counts[0]:
        raise ValueError("adapter_init: pca saw no calibration images")
    explained = []
    for adapter, total, product, count in zip(adapters, sums, outer, counts):
        mean = total / count
        covariance = product / count - torch.outer(mean, mean)
        values, vectors = torch.linalg.eigh(covariance)  # ascending
        values, vectors = values.flip(0)[:width], vectors.flip(1)[:, :width]
        weight = vectors.T / values.clamp_min(eps).sqrt()[:, None]
        adapter.down.weight.copy_(weight.to(adapter.down.weight.dtype))
        adapter.down.bias.copy_((-(weight @ mean)).to(adapter.down.bias.dtype))
        explained.append(float(values.sum() / covariance.trace()))
    return {"tokens_per_block": counts[0], "variance_explained": explained}
