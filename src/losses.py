"""Loss factory.

Default: cross-entropy with `label_smoothing` (that share of the target spread evenly
over all classes).

`neighbour_smoothing: e` (opt-in) moves a share e of the target onto the true class's
likely relatives instead, so the model learns that look-alikes are close rather than that
every wrong class is equally wrong (taxonomy-aware label smoothing):
    target = (1 - label_smoothing - e) * one-hot  +  label_smoothing * uniform
             + e * neighbours(true class)
`neighbour_source` picks the relatives:
  "id"     the classes within `neighbour_window` IDs (default 5 each side), weighted
           exp(-(|d| - 1) / `neighbour_decay`). Class IDs here follow the taxonomy: neighbouring
           IDs' prototypes have cosine 0.80 vs 0.28 for random pairs, and for half the classes
           the most similar class is within 5 IDs (report/progress_log.md, 2026-10-21).
  "proto"  the `neighbour_k` (default 10) classes whose prototypes (`prototype_file`) are most
           similar, weighted softmax(cosine / `neighbour_temperature`); look-alikes from
           other genera count too, and no ID order is assumed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def id_neighbours(num_classes, window, decay):
    """(C, 2*window) neighbour indices and weights; out-of-range slots get weight 0."""
    offsets = torch.tensor([d for d in range(-window, window + 1) if d != 0])
    idx = torch.arange(num_classes)[:, None] + offsets[None, :]
    w = torch.exp(-(offsets.abs().float() - 1) / decay).expand(num_classes, -1).clone()
    valid = (idx >= 0) & (idx < num_classes)
    w = w * valid
    return idx.clamp(0, num_classes - 1), w / w.sum(1, keepdim=True)


def proto_neighbours(prototypes, k, temperature):
    """(C, k) indices and weights of each class's k most similar other prototypes."""
    p = F.normalize(prototypes.float(), dim=1)
    idx, w = [], []
    for start in range(0, p.shape[0], 1024):  # chunks keep memory small for 10,000 classes
        sim = p[start:start + 1024] @ p.T
        rows = torch.arange(start, start + sim.shape[0])
        sim[torch.arange(sim.shape[0]), rows] = -math.inf
        s, i = sim.topk(k, dim=1)
        idx.append(i)
        w.append(torch.softmax(s / temperature, dim=1))
    return torch.cat(idx), torch.cat(w)


class NeighbourSmoothingLoss(nn.Module):
    def __init__(self, idx, weight, neighbour_eps, uniform_eps):
        super().__init__()
        if neighbour_eps < 0 or uniform_eps < 0 or neighbour_eps + uniform_eps >= 1:
            raise ValueError("neighbour_smoothing + label_smoothing must be in [0, 1)")
        self.register_buffer("idx", idx)
        self.register_buffer("weight", weight)
        self.neighbour_eps = float(neighbour_eps)
        self.uniform_eps = float(uniform_eps)

    def forward(self, logits, labels):
        if self.idx.device != logits.device:
            self.to(logits.device)
        logp = F.log_softmax(logits.float(), dim=1)
        hard = -logp.gather(1, labels[:, None]).squeeze(1)
        soft = -(logp.gather(1, self.idx[labels]) * self.weight[labels]).sum(1)
        uniform = -logp.mean(1)
        keep = 1 - self.neighbour_eps - self.uniform_eps
        return (keep * hard + self.neighbour_eps * soft + self.uniform_eps * uniform).mean()


def build_loss(cfg):
    smoothing = float(cfg.get("label_smoothing", 0.0))
    eps = float(cfg.get("neighbour_smoothing", 0.0) or 0.0)
    if not eps:
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    num_classes = int(cfg.num_classes)
    source = str(cfg.get("neighbour_source", "id"))
    if source == "id":
        window, decay = int(cfg.get("neighbour_window", 5)), float(cfg.get("neighbour_decay", 2.0))
        idx, w = id_neighbours(num_classes, window, decay)
        desc = f"IDs within {window}, decay {decay}"
    elif source == "proto":
        blob = torch.load(cfg.prototype_file, map_location="cpu", weights_only=False)
        prototypes = blob["prototypes"] if isinstance(blob, dict) and "prototypes" in blob else blob
        if prototypes.shape[0] != num_classes:
            raise ValueError(f"{cfg.prototype_file} has {prototypes.shape[0]} classes, config has {num_classes}")
        k, temp = int(cfg.get("neighbour_k", 10)), float(cfg.get("neighbour_temperature", 0.05))
        idx, w = proto_neighbours(prototypes, k, temp)
        desc = f"{k} most similar prototypes, temperature {temp}"
    else:
        raise ValueError(f"neighbour_source must be 'id' or 'proto', got {source!r}")
    print(f"[loss] neighbour smoothing {eps} ({desc}) + uniform label smoothing {smoothing}")
    return NeighbourSmoothingLoss(idx, w, eps, smoothing)
