"""Build the ViT-B variants (timm VisionTransformer) used by the CS604 experiments."""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A frozen linear layer plus a small trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return self.base(inputs) + update * self.scaling


class CosineClassifier(nn.Module):
    """Cosine-similarity head suitable for class-prototype initialization."""

    def __init__(self, in_features: int, out_features: int, scale: float = 20.0, force_fp32: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.force_fp32 = force_fp32
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.force_fp32:
            # Match build_prototypes.py: keep backbone AMP, but normalize and
            # compare features outside autocast so nearby class scores don't tie.
            with torch.autocast(device_type=inputs.device.type, enabled=False):
                scale = self.log_scale.float().exp().clamp(max=100.0)
                return scale * F.linear(
                    F.normalize(inputs.float(), dim=-1),
                    F.normalize(self.weight.float(), dim=-1),
                )
        scale = self.log_scale.exp().clamp(max=100.0)
        return scale * F.linear(F.normalize(inputs, dim=-1), F.normalize(self.weight, dim=-1))


class Fp32Linear(nn.Linear):
    """A linear classifier whose logits stay float32 under bf16 autocast.

    timm's plain ``nn.Linear`` head would produce bf16 logits under autocast,
    which ties nearby class scores (as ``cosine_head_fp32`` fixes for the cosine head).
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return F.linear(inputs.float(), self.weight.float(), self.bias.float())


def _classifier(model: nn.Module) -> nn.Module:
    classifier = model.get_classifier()
    if not isinstance(classifier, nn.Module):
        raise TypeError("The timm model does not expose a classifier module")
    return classifier


def _replace_qkv_with_lora(
    model: nn.Module, rank: int, alpha: float, dropout: float
) -> int:
    replaced = 0
    for block in model.blocks:
        qkv = block.attn.qkv
        if not isinstance(qkv, nn.Linear):
            raise TypeError(f"Expected Linear qkv, found {type(qkv).__name__}")
        block.attn.qkv = LoRALinear(qkv, rank=rank, alpha=alpha, dropout=dropout)
        replaced += 1
    return replaced


def build_vit(
    checkpoint: str,
    num_classes: int,
    grad_checkpointing: bool = False,
    mode: str = "full",
    head_type: str = "linear",
    train_head: bool = True,
    last_blocks: int = 1,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    cosine_head_fp32: bool = False,
    image_size: int | None = None,
    cosine_scale: float = 20.0,
    cosine_scale_trainable: bool = True,
    linear_head_fp32: bool = False,
) -> nn.Module:
    # image_size resizes the position embeddings, e.g. DINOv2 (default 518) to 224.
    size_kwargs = {} if image_size is None else {"img_size": int(image_size)}
    model = timm.create_model(checkpoint, pretrained=True, num_classes=num_classes, **size_kwargs)
    if head_type == "cosine":
        old_head = _classifier(model)
        if not isinstance(old_head, nn.Linear):
            raise TypeError("Cosine head replacement expects a linear timm classifier")
        model.head = CosineClassifier(
            old_head.in_features, num_classes, scale=cosine_scale, force_fp32=cosine_head_fp32
        )
    elif head_type == "linear":
        if linear_head_fp32:
            old_head = _classifier(model)
            model.head = Fp32Linear(old_head.in_features, num_classes)
            model.head.load_state_dict(old_head.state_dict())
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    if grad_checkpointing:
        model.set_grad_checkpointing(True)

    if mode == "full":
        model.requires_grad_(True)
    elif mode == "head":
        model.requires_grad_(False)
        _classifier(model).requires_grad_(True)
    elif mode == "last_blocks":
        if not 1 <= last_blocks <= len(model.blocks):
            raise ValueError(f"last_blocks must be in [1, {len(model.blocks)}]")
        model.requires_grad_(False)
        for block in model.blocks[-last_blocks:]:
            block.requires_grad_(True)
        model.norm.requires_grad_(True)
        _classifier(model).requires_grad_(train_head)
    elif mode == "lora":
        model.requires_grad_(False)
        _replace_qkv_with_lora(model, lora_rank, lora_alpha, lora_dropout)
        _classifier(model).requires_grad_(train_head)
    else:
        raise ValueError(f"Unknown adaptation mode: {mode}")
    _freeze_scale_unless_trainable(_classifier(model), cosine_scale_trainable)
    return model


def _freeze_scale_unless_trainable(classifier: nn.Module, trainable: bool) -> None:
    """``cosine_scale_trainable: false`` keeps the cosine temperature at its initial value."""

    if trainable:
        return
    if not isinstance(classifier, CosineClassifier):
        raise ValueError("cosine_scale_trainable: false needs head_type: cosine")
    classifier.log_scale.requires_grad_(False)


class FeatureHead(nn.Module):
    """The classifier alone, for training on cached pre-logits features.

    Built by the same code as the full model's head (``head_type``, ``cosine_scale``,
    ``cosine_head_fp32``, ``cosine_scale_trainable``, ``linear_head_fp32``), and it exposes
    ``get_classifier()``, so ``apply_class_prototypes`` and
    ``optimizer_param_groups`` treat it exactly like the full model in ``mode: head``.
    The state dict keys (``head.*``) match the full model's classifier keys.
    """

    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def get_classifier(self) -> nn.Module:
        return self.head

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


def build_feature_head(cfg: dict, in_features: int) -> FeatureHead:
    """A trainable head matching ``build_from_config(cfg)``'s classifier in ``mode: head``."""

    head_type = str(cfg.get("head_type", "linear"))
    num_classes = int(cfg["num_classes"])
    if head_type == "cosine":
        head: nn.Module = CosineClassifier(
            in_features,
            num_classes,
            scale=float(cfg.get("cosine_scale", 20.0)),
            force_fp32=bool(cfg.get("cosine_head_fp32", False)),
        )
    elif head_type == "linear":
        linear = Fp32Linear if bool(cfg.get("linear_head_fp32", False)) else nn.Linear
        head = linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")
    _freeze_scale_unless_trainable(head, bool(cfg.get("cosine_scale_trainable", True)))
    return FeatureHead(head)


def apply_class_prototypes(model: nn.Module, prototypes: torch.Tensor) -> None:
    """Copy one normalized feature average into each class weight."""

    classifier = _classifier(model)
    weight = getattr(classifier, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("The classifier has no weight parameter")
    if tuple(weight.shape) != tuple(prototypes.shape):
        raise ValueError(
            f"Prototype shape {tuple(prototypes.shape)} does not match "
            f"classifier weight {tuple(weight.shape)}"
        )
    normalized = F.normalize(prototypes.float(), dim=1).to(
        device=weight.device, dtype=weight.dtype
    )
    with torch.no_grad():
        weight.copy_(normalized)
        bias = getattr(classifier, "bias", None)
        if isinstance(bias, torch.Tensor):
            bias.zero_()


def build_from_config(cfg: dict) -> nn.Module:
    """Build a model from the shared YAML configuration dictionary."""

    return build_vit(
        checkpoint=cfg["timm_checkpoint"],
        num_classes=int(cfg["num_classes"]),
        grad_checkpointing=bool(cfg.get("grad_checkpointing", False)),
        mode=str(cfg.get("mode", "full")),
        head_type=str(cfg.get("head_type", "linear")),
        train_head=bool(cfg.get("train_head", True)),
        last_blocks=int(cfg.get("last_blocks", 1)),
        lora_rank=int(cfg.get("lora_rank", 8)),
        lora_alpha=float(cfg.get("lora_alpha", 16.0)),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        cosine_head_fp32=bool(cfg.get("cosine_head_fp32", False)),
        image_size=None if cfg.get("image_size") is None else int(cfg["image_size"]),
        cosine_scale=float(cfg.get("cosine_scale", 20.0)),
        cosine_scale_trainable=bool(cfg.get("cosine_scale_trainable", True)),
        linear_head_fp32=bool(cfg.get("linear_head_fp32", False)),
    )


def model_data_config(model: nn.Module) -> dict:
    """timm's preprocessing for ``model``, at the input size it was built for.

    timm keeps the checkpoint's default input size in the data config even when
    ``img_size`` changed it (DINOv2: 518), so take the size from the patch embedding.
    """

    from timm.data import resolve_model_data_config

    config = resolve_model_data_config(model)
    img_size = getattr(getattr(model, "patch_embed", None), "img_size", None)
    if img_size is not None:
        config["input_size"] = (config["input_size"][0], *img_size)
    return config


def use_compact_checkpoints(cfg: dict) -> bool:
    """Compact checkpoints drop frozen backbone weights that timm can rebuild.

    Safe only when the frozen backbone is exactly the pretrained one, so it is the
    default for LoRA/head runs without ``init_checkpoint``.
    """

    frozen_backbone = str(cfg.get("mode", "full")) in ("lora", "head")
    default = frozen_backbone and not cfg.get("init_checkpoint")
    compact = bool(cfg.get("compact_checkpoints", default))
    if compact and not default:
        raise ValueError(
            "compact_checkpoints needs mode lora/head and no init_checkpoint; "
            "set compact_checkpoints: false"
        )
    return compact


def _compact_keys(model: nn.Module) -> set[str]:
    classifier_ids = {id(p) for p in _classifier(model).parameters()}
    return {
        name for name, p in model.named_parameters()
        if p.requires_grad or id(p) in classifier_ids
    } | {name for name, _ in model.named_buffers()}


def base_fingerprint(model: nn.Module) -> str:
    """Hash a sample of frozen backbone tensors to catch a different pretrained base."""

    import hashlib

    keep = _compact_keys(model)
    state = model.state_dict()
    frozen = sorted(name for name in state if name not in keep)
    digest = hashlib.sha256()
    for name in sorted({frozen[0], frozen[len(frozen) // 2], frozen[-1]} if frozen else ()):
        digest.update(name.encode())
        digest.update(state[name].detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_model_state(model: nn.Module, compact: bool) -> dict:
    """The ``model`` entry of a checkpoint, plus its format fields."""

    state = model.state_dict()
    if not compact:
        return {"model": state, "model_format": "full"}
    keep = _compact_keys(model)
    return {
        "model": {name: tensor for name, tensor in state.items() if name in keep},
        "model_format": "compact",
        "base_fingerprint": base_fingerprint(model),
    }


def load_model_state(model: nn.Module, checkpoint: dict) -> None:
    """Load a full or compact checkpoint into a model built from its config."""

    state = checkpoint["model"]
    if checkpoint.get("model_format", "full") == "full":
        model.load_state_dict(state)
        return
    expected = _compact_keys(model)
    if set(state) != expected:
        raise ValueError(
            "Compact checkpoint keys do not match this model's trainable/classifier "
            f"weights ({len(set(state) ^ expected)} differ)"
        )
    if checkpoint.get("base_fingerprint") != base_fingerprint(model):
        raise ValueError(
            "Frozen backbone differs from the one this compact checkpoint was trained on "
            "(timm weights changed?)"
        )
    model.load_state_dict(state, strict=False)


def optimizer_param_groups(
    model: nn.Module, lr: float, head_lr: float | None = None
) -> list[dict]:
    """AdamW parameter groups: one group at ``lr``, or backbone and head groups.

    Without ``head_lr`` this is a single group of every trainable parameter, as
    before. With it, the trainable classifier parameters (cosine weight and
    temperature) get ``head_lr``. The backbone group stays first, so
    ``param_groups[0]["lr"]`` is still the backbone rate in the metrics log.
    """

    trainable = [p for p in model.parameters() if p.requires_grad]
    if head_lr is None:
        return [{"params": trainable, "lr": lr}]
    if head_lr <= 0:
        raise ValueError(f"head_lr must be positive, got {head_lr}")
    classifier_ids = {id(p) for p in _classifier(model).parameters()}
    head = [p for p in trainable if id(p) in classifier_ids]
    backbone = [p for p in trainable if id(p) not in classifier_ids]
    if not head:
        raise ValueError("head_lr is set but the classifier is frozen; set train_head: true")
    groups = [{"params": backbone, "lr": lr}] if backbone else []
    groups.append({"params": head, "lr": head_lr})
    return groups


def parameter_report(model: nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    classifier_ids = {id(p) for p in _classifier(model).parameters()}
    classifier_total = sum(p.numel() for p in model.parameters() if id(p) in classifier_ids)
    classifier_trainable = sum(
        p.numel() for p in model.parameters() if id(p) in classifier_ids and p.requires_grad
    )
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
        "classifier_params": classifier_total,
        "classifier_trainable_params": classifier_trainable,
        "backbone_trainable_params": trainable - classifier_trainable,
    }
