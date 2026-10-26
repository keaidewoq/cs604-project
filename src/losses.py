"""Loss factory."""

import torch.nn as nn


def build_loss(cfg, model=None):
    smoothing = float(cfg.get("label_smoothing", 0.0))
    if cfg.get("cosine_margin"):  # opt-in (Van): CosFace margin on the cosine head
        from src.van_losses import CosineMarginLoss

        return CosineMarginLoss(model.get_classifier(), float(cfg.cosine_margin), smoothing)
    return nn.CrossEntropyLoss(label_smoothing=smoothing)
