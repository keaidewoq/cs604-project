"""Input-space visual prompts composed with LoRA on a CLS-pooled timm ViT.

Hooks preserve the original block names and checkpoint format. Shallow prompts
propagate through every block; deep prompts replace their predecessor's outputs.
"""

from functools import partial

import torch
from torch import nn


class VisualPrompts(nn.Module):
    def __init__(self, dim, depth, length, mode, patch_size, reference):
        super().__init__()
        if mode not in ("shallow", "deep") or length <= 0:
            raise ValueError("VPT requires shallow/deep mode and positive prompt_length")
        self.length = length
        self.mode = mode
        layers = depth if mode == "deep" else 1
        self.embeddings = nn.Parameter(reference.new_empty(layers, length, dim))
        # VPT's patch-size-aware uniform initialization. Preserve the training
        # RNG so prompt length does not change classifier init or data order.
        bound = (6 / (3 * patch_size[0] * patch_size[1] + dim)) ** 0.5
        with torch.random.fork_rng(devices=[]):
            nn.init.uniform_(self.embeddings, -bound, bound)
        self.embeddings._no_weight_decay = True

    def before_block(self, block, args, *, index):
        x = args[0]
        prompt = self.embeddings[index].to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        patches = x[:, 1:] if index == 0 else x[:, 1 + self.length:]
        return (torch.cat((x[:, :1], prompt, patches), dim=1), *args[1:])


def apply_vpt(model, cfg):
    mode = cfg.get("prompt_mode", "none")
    length = int(cfg.get("prompt_length", 0))
    if mode == "none":
        if length:
            raise ValueError("prompt_mode=none requires prompt_length=0")
        return model
    if getattr(model, "global_pool", None) != "token" or model.num_prefix_tokens != 1:
        raise ValueError("VPT currently supports CLS-pooled ViTs with one CLS token")
    if getattr(model, "grad_checkpointing", False):
        raise ValueError("VPT hooks do not support timm gradient checkpointing")
    if hasattr(model, "vpt"):
        raise ValueError("VPT is already attached")
    model.vpt = VisualPrompts(
        model.num_features, len(model.blocks), length, mode,
        model.patch_embed.patch_size, model.cls_token,
    )
    # In this backbone norm_pre is Identity. Prompts enter after positional
    # embedding and dropout; the original CLS/patch positions remain intact.
    for index, block in enumerate(model.blocks):
        if index == 0 or mode == "deep":
            block.register_forward_pre_hook(partial(model.vpt.before_block, index=index))
    return model
