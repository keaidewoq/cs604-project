"""Train chosen pieces of the pretrained backbone itself, alone or next to LoRA.

Two ways to choose (both set in the config, both work with any PEFT method):

`unfreeze_modules: [blocks.0.attn.qkv, ...]` (or `unfreeze_modules_file`, a YAML
list written by scripts/select_params.py) fully trains those Linear / conv
layers: surgical fine-tuning (Lee et al., ICLR 2023) when chosen by depth,
partial fine-tuning (Ye et al., 2023) when chosen by fine-tuned angle. LoRA skips
layers listed here, since they are already trained at full rank.

`sparse_mask_file: mask.pt` ({param name: bool tensor}) trains only the masked
entries of those weights: GPS (Zhang et al., CVPR 2024), which keeps the top-K
input connections of every output neuron by gradient magnitude. The gradient is
multiplied by the mask, and weight decay is off for these tensors, so AdamW
leaves unmasked entries exactly at their pretrained values. Only masked entries
count as trainable.

`unfreeze_lr` gives these pretrained weights their own learning rate (full FT
pilots found 1e-5 to 3e-5 best, far below LoRA's 3e-4). `method: selected`
trains only these pieces (plus `unfreeze_extra`), with no LoRA.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml


def resolve_unfreeze_modules(cfg):
    """The module names to fully train: `unfreeze_modules` plus `unfreeze_modules_file`."""
    names = list(cfg.get("unfreeze_modules") or [])
    path = cfg.get("unfreeze_modules_file")
    if path:
        loaded = yaml.safe_load(Path(path).read_text()) or []
        if not isinstance(loaded, list):
            raise ValueError(f"{path} must hold a YAML list of module names")
        names += [str(n) for n in loaded]
    return list(dict.fromkeys(names))


def _base_layer(module):
    # LoRALinear keeps the pretrained layer as .linear
    return getattr(module, "linear", module)


def unfreeze_modules(model, names):
    n = 0
    for name in names:
        layer = _base_layer(model.get_submodule(name))
        if not isinstance(layer, (nn.Linear, nn.Conv2d)):
            raise TypeError(f"unfreeze_modules: {name} is {type(layer).__name__}, expected Linear or Conv2d")
        layer.weight.requires_grad = True
        layer.weight._unfrozen = True  # gets unfreeze_lr
        n += layer.weight.numel()
        if layer.bias is not None:
            layer.bias.requires_grad = True
            layer.bias._no_weight_decay = True
            n += layer.bias.numel()
    print(f"[selected] fully training {len(names)} layers: {n:,} params")


def apply_sparse_mask(model, path):
    masks = torch.load(path, map_location="cpu")
    params = dict(model.named_parameters())
    n = 0
    for name, mask in masks.items():
        if name not in params:
            raise KeyError(f"sparse_mask_file: no parameter {name!r} in the model")
        param = params[name]
        if tuple(mask.shape) != tuple(param.shape):
            raise ValueError(f"sparse mask {name}: shape {tuple(mask.shape)} != {tuple(param.shape)}")
        owner = model.get_submodule(name.rpartition(".")[0])
        buffer = "gps_mask_" + name.rpartition(".")[2]
        # a buffer, so .to(device) moves it with the model
        owner.register_buffer(buffer, mask.to(param.dtype), persistent=False)
        param.requires_grad = True
        param.register_hook(lambda grad, owner=owner, buffer=buffer: grad * getattr(owner, buffer))
        param._unfrozen = True
        param._no_weight_decay = True  # decay would move the unmasked entries too
        param._trainable_count = int(mask.sum())
        n += param._trainable_count
    print(f"[selected] sparse training {n:,} masked entries in {len(masks)} weights ({path})")


def apply_selected(model, cfg):
    """`method: selected`: no new modules; apply_method unfreezes what the config chose."""
    return model
