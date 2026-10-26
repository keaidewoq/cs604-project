"""Van's loss option: a CosFace margin for the cosine head (`cosine_margin`).

During training the target class's logit becomes s * (cos - m) instead of s * cos,
so a class must beat the others by a margin m. s is the head's learned temperature,
detached in the margin term so the margin does not push it. In eval mode (validation)
this is plain cross-entropy. Tuned on Van's laptop holdout: m = 0.1 added +0.6 points
at 1 epoch; m = 0.2 lost 0.5.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class CosineMarginLoss(nn.Module):
    def __init__(self, head, margin, label_smoothing=0.0):
        super().__init__()
        if not hasattr(head, "log_scale"):
            raise ValueError("cosine_margin needs head_type: cosine")
        self.head = [head]  # a list, so the head is not registered as a submodule
        self.margin = float(margin)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, labels):
        head = self.head[0]
        if head.training and self.margin:
            scale = head.log_scale.detach().float().exp().clamp(max=100.0)
            target = F.one_hot(labels, logits.shape[1]).to(logits.dtype)
            logits = logits - (self.margin * scale) * target
        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
