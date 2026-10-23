"""Exercise official AdaLoRA training, time allocation, checkpointing and launchers."""
import copy
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from timm.models.vision_transformer import VisionTransformer

from src.heads import replace_head
from src.model import apply_method, build_model
from src.qq_adalora_ablation import arm_config, plan_sweep, save_sweep_summary
from src.utils import load_config, set_seed


def tiny_backbone(classes=10):
    return VisionTransformer(img_size=32, patch_size=16, embed_dim=24, depth=2,
                             num_heads=3, mlp_ratio=2, num_classes=classes)


def tiny_config():
    cfg = load_config('configs/qq_adalora_ablation.yaml')
    cfg.update(dict(method='adalora', pretrained=False, img_size=32, num_classes=10,
                    adalora_init_rank=4, adalora_target_rank=2, lora_alpha=4,
                    max_adaptation_pct=None, adalora_mask_interval=1, bf16=False))
    return cfg


def test_real_vit_sweep_counts_initial_parameters_under_cap():
    torch.set_num_threads(1)
    cfg = load_config('configs/qq_adalora_ablation.yaml')
    arms = plan_sweep(cfg)
    assert [arm['arm'] for arm in arms] == ['lora_r24', 'adalora_t8', 'adalora_t16', 'adalora_t20']
    assert arms[0]['adaptation_parameters'] == 4264704
    assert all(arm['adaptation_parameters'] == 4265856 for arm in arms[1:])
    assert all(arm['adaptation_pct'] < 5 for arm in arms)
    assert [arm['target_components'] for arm in arms[1:]] == [384, 768, 960]
    assert cfg.pretrained and cfg.method == 'lora'


def test_official_adalora_updates_masks_freezes_and_reloads(tmp_path):
    from src.train import save_checkpoint
    from src.predict import load_checkpoint
    torch.set_num_threads(1)
    set_seed(42)
    cfg = tiny_config()
    base = tiny_backbone()
    replace_head(base, cfg)
    initial = copy.deepcopy(base.state_dict())
    images, labels = torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 2])
    base.eval()
    before = base(images).detach()
    model = apply_method(base, cfg)
    torch.testing.assert_close(model(images), before)
    controller = model.adalora_controller
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for step, fraction in enumerate([.05, .15, .3, .45, .6, .8, .9, 1.], start=1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(images), labels).backward()
        regularizer = controller.regularization()
        assert torch.isfinite(regularizer) and regularizer > 0
        regularizer.backward()
        optimizer.step()
        controller.after_step(step, fraction)
        if step == 1:
            assert controller.allocator.exp_avg_ipt
        if fraction < .1:
            assert controller.snapshot()['active_components'] == 32
        if fraction >= .8:
            assert controller.frozen
            if fraction == .8:
                frozen = copy.deepcopy(controller.pattern)
            assert controller.pattern == frozen
            for name, layer in controller.layers:
                mask = torch.tensor(controller.pattern[f'{name}.lora_E.default'])
                assert (layer.lora_E['default'].detach().view(-1)[~mask] == 0).all()
    snapshot = controller.snapshot()
    assert 0 < snapshot['active_components'] <= 16
    assert snapshot['active_parameters'] < snapshot['allocated_parameters']
    assert snapshot['average_active_rank'] <= 2
    with pytest.raises(ValueError, match='increasing'):
        controller.after_step(8, 1.)
    model.eval()
    expected = model(images).detach()
    path = tmp_path / 'last.pt'
    save_checkpoint(path, model, cfg, 1, {'val_acc': .5})
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint['trainable_only']
    assert checkpoint['adalora']['frozen']
    assert any('lora_E.default' in key for key in checkpoint['state_dict'])
    assert not any('ranknum' in key for key in checkpoint['state_dict'])
    assert not any('base_layer.weight' in key for key in checkpoint['state_dict'])
    restored = tiny_backbone()
    replace_head(restored, cfg)
    restored.load_state_dict(initial)
    restored = apply_method(restored, cfg)
    load_checkpoint(restored, path, torch.device('cpu'))
    restored.eval()
    torch.testing.assert_close(restored(images), expected)


def test_rank_schedule_has_initial_cubic_and_final_phases():
    cfg = tiny_config()
    model = tiny_backbone()
    replace_head(model, cfg)
    controller = apply_method(model, cfg).adalora_controller
    controller.progress = .05
    assert controller.allocator.budget_schedule(10) == (32, False)
    controller.progress = .45
    middle, mask = controller.allocator.budget_schedule(10)
    assert 16 < middle < 32 and mask
    controller.progress = .8
    assert controller.allocator.budget_schedule(10) == (16, True)


def test_four_arm_runner_uses_real_trainer_and_summary_on_tiny_images(tmp_path, monkeypatch):
    import src.model as models
    from src.qq_lora_ablation import prepare_split, run_arm, save_results
    torch.set_num_threads(1)
    source = tmp_path / 'data'
    (source / 'images').mkdir(parents=True)
    for label in range(2):
        for index in range(8):
            Image.new('RGB', (40, 32), (220, 50, 30) if label else (30, 50, 220)).save(
                source / f'images/{label}_{index}.png')
    (source / 'train.txt').write_text(''.join(
        f'images/{label}_{index}.png {label}\n' for label in range(2) for index in range(8)))
    monkeypatch.setattr(models.timm, 'create_model', lambda *args, **kwargs: tiny_backbone(kwargs['num_classes']))
    cfg = load_config('configs/qq_adalora_ablation.yaml')
    cfg.update(dict(img_size=32, num_classes=10, ablation_classes=2, ablation_holdout=2,
                    num_workers=0, batch_size=4, grad_accum_steps=1, max_train_minutes=.006,
                    max_adaptation_pct=None, bf16=False, adalora_mask_interval=1))
    split = prepare_split(source, tmp_path / 'local', cfg)
    root = tmp_path / 'results'
    root.mkdir()
    arms = [dict(arm='lora_r24', method='lora', target_rank=0)] + [
        dict(arm=f'adalora_t{rank}', method='adalora', target_rank=rank) for rank in [8, 16, 20]]
    results = []
    for arm in arms:
        arm_cfg = arm_config(cfg, arm)
        result = run_arm(arm_cfg, split, root, arm['arm'], 32, .25)
        assert result['training_minutes'] >= cfg.max_train_minutes
        assert result['images_per_second'] > 0
        assert result == run_arm(arm_cfg, split, root, arm['arm'], 32, .25)
        final = None
        if arm['method'] == 'adalora':
            final = json.loads((Path(result['run_dir']) / 'adalora_final.json').read_text())
            assert final['frozen']
            assert final['average_active_rank'] <= arm['target_rank']
            assert (Path(result['run_dir']) / 'adalora_rank_log.csv').is_file()
        results.append(dict(**result, seed=42, method=arm['method'], target_rank=arm['target_rank'],
                            final_average_rank=final['average_active_rank'] if final else 24.,
                            final_active_pct=final['active_adaptation_pct'] if final else result['adaptation_pct'],
                            allocation_frozen=final['frozen'] if final else True))
        save_results(root, results)
        save_sweep_summary(root, results)
    summary = json.loads((root / 'comparison.json').read_text())
    assert summary['complete'] and summary['allocation_complete']
    assert len(summary['comparisons']) == 3
    import matplotlib
    matplotlib.use('Agg')
    book = json.loads(Path('notebooks/colab_launcher_qq_adalora.ipynb').read_text())
    summary_cell = next(''.join(c['source']) for c in book['cells']
                        if c['cell_type'] == 'code' and 'comparison = json.loads' in ''.join(c['source']))
    exec(summary_cell, {'EXPERIMENT_DIR': root, 'display': lambda value: None})
    assert (root / 'comparison.png').is_file()


def test_adalora_memory_probe_reaches_allocation_before_timed_runs(monkeypatch):
    import sys
    from types import SimpleNamespace
    import src.qq_adalora_ablation as sweep
    base = load_config('configs/qq_adalora_ablation.yaml')
    arms = [dict(arm='adalora_t8', method='adalora', target_rank=8, adaptation_parameters=4265856,
                 original_backbone_parameters=86250240, adaptation_pct=4.945906,
                 initial_components=1152, target_components=384)]
    monkeypatch.setattr(sweep, 'plan_sweep', lambda cfg: [dict(**arms[0], unused=True)] + arms)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda index: SimpleNamespace(
        name='NVIDIA A100', total_memory=40 * 1024**3))
    monkeypatch.setattr(sys, 'argv', ['sweep', '--memory-check-only'])
    seen = []
    def probe(cfg, arm):
        seen.append(cfg)
        assert cfg.method == 'adalora' and cfg.adalora_target_rank == 8
        assert cfg.batch_size == 64 and cfg.grad_accum_steps == 4
    monkeypatch.setattr(sweep, 'check_training_memory', probe)
    monkeypatch.setattr(sweep, 'run_arm', lambda *args: pytest.fail('Timed arm started'))
    sweep.main()
    assert len(seen) == 1 and base.seed == 42


def test_memory_probe_runs_actual_adalora_regularization_and_allocation(monkeypatch):
    from contextlib import nullcontext
    import src.qq_lora_dora_ablation as probe
    cfg = tiny_config()
    cfg.update(dict(batch_size=2, grad_accum_steps=4))
    base = tiny_backbone()
    replace_head(base, cfg)
    model = apply_method(base, cfg)
    monkeypatch.setattr(model, 'to', lambda device: model)
    monkeypatch.setattr(probe, 'build_model', lambda config: model)
    monkeypatch.setattr(probe, 'amp_context', lambda *args: nullcontext())
    randn, arange = torch.randn, torch.arange
    monkeypatch.setattr(torch, 'randn', lambda *args, **kwargs: randn(*args))
    monkeypatch.setattr(torch, 'arange', lambda *args, **kwargs: arange(*args))
    for name in ['empty_cache', 'reset_peak_memory_stats', 'synchronize']:
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 1024**3)
    # Run the real probe's model/loss/AdamW/allocation path with CPU tensors.
    assert probe.check_training_memory(cfg, dict(arm='adalora_t2', rank=4, dora=False)) == 1.
    final = model.adalora_controller.snapshot()
    assert final['frozen'] and final['step'] == 2
    assert 0 < final['active_components'] <= 16


def test_launcher_compiles_and_installs_adalora_requirements():
    book = json.loads(Path('notebooks/colab_launcher_qq_adalora.ipynb').read_text())
    text = ''
    for index, cell in enumerate(book['cells']):
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code':
            compile(source, f'adalora_cell_{index}', 'exec')
            text += source
    assert 'requirements-adalora.txt' in text
    assert 'src.qq_adalora_ablation' in text
    assert 'qq_lora_ablation' in text
