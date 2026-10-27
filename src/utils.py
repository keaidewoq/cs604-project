"""Shared helpers: config merge, seeding, param counts, timing, VRAM."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PEFT_METHODS = {"lora", "adapter", "adaptformer", "bitfit", "last_blocks"}


class Config:
    """Thin attribute-access wrapper around a flat dict."""

    def __init__(self, mapping=None):
        object.__setattr__(self, "_data", dict(mapping or {}))

    def __getattr__(self, name):
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name == "_data":
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __contains__(self, name):
        return name in self._data

    def get(self, name, default=None):
        return self._data.get(name, default)

    def to_dict(self):
        return dict(self._data)

    def update(self, other):
        if isinstance(other, Config):
            other = other.to_dict()
        self._data.update(other)
        return self


def load_config(path, _seen=None):
    """Load a YAML config, recursively merging `inherits:` parents."""
    path = Path(path).resolve()
    seen = _seen if _seen is not None else set()
    if path in seen:
        raise ValueError(f"Circular config inherit: {path}")
    seen.add(path)

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    inherits = raw.pop("inherits", None)
    if inherits:
        parent = load_config(path.parent / inherits, _seen=seen)
        parent.update(raw)
        return parent
    return Config(raw)


def apply_overrides(cfg, **kwargs):
    """Set any non-None kwargs on cfg (CLI overrides)."""
    for key, value in kwargs.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def save_config(cfg, dest):
    dest = Path(dest)
    with dest.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """Return (trainable, total, percent_trainable)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(getattr(p, "_trainable_count", p.numel()) for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    return trainable, total, pct


def format_params(n):
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.start


class VRAMTracker:
    """Wraps torch.cuda.max_memory_allocated. Returns 0 on CPU."""

    def reset(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def peak_bytes(self):
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.max_memory_allocated())

    def peak_gb(self):
        return self.peak_bytes() / (1024 ** 3)


def gpu_count():
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def gpu_hours(elapsed_s):
    return (elapsed_s / 3600.0) * gpu_count()


def log_run_stats(tag, model, timer, vram):
    trainable, total, pct = count_parameters(model)
    elapsed = timer.elapsed()
    print(
        f"[{tag}] trainable={format_params(trainable)}/{format_params(total)} "
        f"({pct:.2f}%) wall_s={elapsed:.1f} gpu_hours={gpu_hours(elapsed):.4f} "
        f"peak_vram_gb={vram.peak_gb():.3f}"
    )
    return {
        "tag": tag,
        "trainable": trainable,
        "total": total,
        "trainable_pct": pct,
        "wall_s": elapsed,
        "gpu_hours": gpu_hours(elapsed),
        "peak_vram_gb": vram.peak_gb(),
    }


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
