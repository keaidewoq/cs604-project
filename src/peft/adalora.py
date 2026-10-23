"""PEFT AdaLoRA layers/importance scoring with a training-time budget schedule.

Uses Hugging Face PEFT's SVDLinear, RankAllocator and orthogonal regularizer
formula. Only the cubic budget schedule uses elapsed training fraction instead
of a guessed number of optimizer steps. See docs/adalora_experiment.md.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.peft.lora import LORA_TARGETS


def _peft_api():
    try:
        from peft import AdaLoraConfig
        from peft.tuners.adalora.layer import RankAllocator, SVDLinear
    except ImportError as exc:
        raise ImportError('AdaLoRA needs: pip install -r requirements-adalora.txt') from exc
    return AdaLoraConfig, RankAllocator, SVDLinear


class AdaLoraController:
    def __init__(self, model, layers, cfg):
        AdaLoraConfig, RankAllocator, _ = _peft_api()
        initial = int(cfg.get('adalora_init_rank', 24))
        target = int(cfg.get('adalora_target_rank', 16))
        self.initial_fraction = float(cfg.get('adalora_initial_fraction', .10))
        self.final_fraction = float(cfg.get('adalora_final_fraction', .20))
        self.interval = int(cfg.get('adalora_mask_interval', 10))
        self.orth_weight = float(cfg.get('adalora_orth_reg_weight', .5))
        if not 0 < target < initial:
            raise ValueError('AdaLoRA needs 0 < target rank < initial rank')
        if not (0 <= self.initial_fraction < 1 - self.final_fraction < 1):
            raise ValueError('AdaLoRA needs an initial phase, pruning phase and positive final phase')
        if self.interval < 1 or not math.isfinite(self.orth_weight) or self.orth_weight < 0:
            raise ValueError('Use a positive mask interval and finite nonnegative orthogonal weight')
        self.model, self.layers = model, layers
        self.frozen, self.pattern, self.progress = False, None, 0.
        self.initial_rank, self.target_rank = initial, target
        self.initial_budget, self.target_budget = initial * len(layers), target * len(layers)
        self.budget = self.initial_budget
        self.last_step = 0

        controller = self
        class TimeRankAllocator(RankAllocator):
            def budget_schedule(self, step):
                progress = controller.progress
                if progress <= controller.initial_fraction:
                    return self.init_bgt, False
                if progress >= 1 - controller.final_fraction:
                    return self.target_bgt, True
                fraction = ((progress - controller.initial_fraction) /
                            (1 - controller.final_fraction - controller.initial_fraction))
                budget = int(self.target_bgt + (self.init_bgt - self.target_bgt) * (1 - fraction)**3)
                return budget, budget < self.init_bgt and step % controller.interval == 0

        # Importance updates run every real optimizer step. TimeRankAllocator
        # controls pruning; the large step limit avoids the stock step cutoff.
        peft_cfg = AdaLoraConfig(init_r=initial, target_r=target, total_step=10**12,
                                tinit=0, tfinal=0, deltaT=self.interval,
                                beta1=float(cfg.get('adalora_beta1', .85)),
                                beta2=float(cfg.get('adalora_beta2', .85)))
        self.allocator = TimeRankAllocator(model, peft_cfg, 'default')

    def regularization(self):
        """Same averaged Frobenius Gram penalty as PEFT; evaluate in FP32."""
        if self.orth_weight == 0:
            return None
        penalties = []
        for _, layer in self.layers:
            for matrix, left in [(layer.lora_A['default'], True), (layer.lora_B['default'], False)]:
                with torch.autocast(device_type=matrix.device.type, enabled=False):
                    value = matrix.float()
                    gram = value @ value.T if left else value.T @ value
                    penalties.append(torch.linalg.matrix_norm(
                        gram - torch.eye(gram.shape[0], device=gram.device), ord='fro'))
        return self.orth_weight * torch.stack(penalties).mean()

    def after_step(self, step, progress):
        """Call after optimizer.step, before clearing accumulated gradients."""
        if step <= self.last_step or not math.isfinite(progress):
            raise ValueError('AdaLoRA needs increasing optimizer steps and finite progress')
        self.last_step = step
        self.progress = max(self.progress, min(max(progress, 0.), 1.))
        if self.frozen:
            self.allocator.mask_using_rank_pattern(self.model, self.pattern)
            return
        self.budget, pattern = self.allocator.update_and_allocate(self.model, step)
        if pattern is not None:
            self.pattern = pattern
        if self.progress >= 1 - self.final_fraction:
            self.frozen = True
            self.allocator.reset_ipt()

    def snapshot(self):
        ranks = {}
        full_adapter = active_adapter = 0
        for name, layer in self.layers:
            key = f'{name}.lora_E.default'
            kept = sum(self.pattern[key]) if self.pattern is not None else self.initial_rank
            ranks[name] = kept
            unit = layer.in_features + layer.out_features + 1
            full_adapter += self.initial_rank * unit
            active_adapter += kept * unit
        stats = getattr(self.model, 'adaptation_stats', {})
        allocated = stats.get('adaptation_parameters', full_adapter)
        total = stats.get('original_backbone_parameters', 0)
        active = allocated - full_adapter + active_adapter
        return dict(step=self.last_step, progress=self.progress, frozen=self.frozen,
                    scheduled_rank_budget=self.budget, target_rank_budget=self.target_budget,
                    active_components=sum(ranks.values()), average_active_rank=sum(ranks.values()) / len(ranks),
                    allocated_parameters=allocated, active_parameters=active,
                    active_adaptation_pct=100 * active / total if total else 0., ranks=ranks)


def apply_adalora(model, cfg):
    AdaLoraConfig, _, SVDLinear = _peft_api()
    initial = int(cfg.get('adalora_init_rank', 24))
    targets = list(cfg.get('lora_targets', ['qkv']))
    if not targets or any(name not in LORA_TARGETS for name in targets):
        raise ValueError(f'AdaLoRA targets must be selected from {list(LORA_TARGETS)}')
    if cfg.get('lora_dora', False) or cfg.get('prompt_mode', 'none') != 'none':
        raise ValueError('This AdaLoRA comparison does not combine DoRA or visual prompts')
    layer_cfg = AdaLoraConfig(init_r=initial, target_r=int(cfg.get('adalora_target_rank', 16)),
                             total_step=10**12, lora_dropout=float(cfg.get('lora_dropout', 0.)))
    layers = []
    for index, block in enumerate(model.blocks):
        for name in targets:
            parent = getattr(block, LORA_TARGETS[name])
            base = getattr(parent, name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f'AdaLoRA expects Linear at blocks.{index}.{LORA_TARGETS[name]}.{name}')
            layer = SVDLinear(base, 'default', config=layer_cfg, r=initial,
                              lora_alpha=float(cfg.get('lora_alpha', initial)))
            setattr(parent, name, layer)
            layers.append((f'blocks.{index}.{LORA_TARGETS[name]}.{name}', layer))
    model.adalora_controller = AdaLoraController(model, layers, cfg)
    print(f'[adalora] {len(layers)} SVD layers, init rank {initial}, '
          f'target average {cfg.get("adalora_target_rank", 16)}')
    return model
