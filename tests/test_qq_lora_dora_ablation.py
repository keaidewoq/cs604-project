"""Check DoRA updates, real ViT budget accounting, and controlled experiment output."""
import csv
import json
from pathlib import Path

import pytest
import torch

from src.model import build_model
from src.peft.lora import LoRALinear
from src.qq_lora_dora_ablation import matched_epoch_results, plan_arms, save_comparison
from src.utils import Config, load_config


def test_cuda_hardware_metadata_can_be_saved_before_training(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import yaml
    import src.qq_lora_dora_ablation as experiment
    from src.utils import save_config

    source = tmp_path / 'data'
    source.mkdir()
    (source / 'train.txt').write_text(''.join(
        f'images/{label}_{index}.png {label}\n' for label in range(2) for index in range(6)))
    output = tmp_path / 'results'
    monkeypatch.setattr(sys, 'argv', ['comparison', '--data_root', str(source),
                                     '--output_dir', str(output), '--classes', '2',
                                     '--split_dir', str(tmp_path / 'local_split')])
    monkeypatch.setattr(torch, '__version__', torch.torch_version.TorchVersion('2.11.0+cu130'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(
        name='NVIDIA A100-SXM4-40GB', total_memory=40 * 1024**3))
    monkeypatch.setattr(experiment, 'plan_arms', lambda cfg: [dict(
        arm='lora_r24', rank=24, dora=False, adaptation_parameters=4264704,
        original_backbone_parameters=86250240, adaptation_pct=4.944571), dict(
        arm='dora_r23', rank=23, dora=True, adaptation_parameters=4200192,
        original_backbone_parameters=86250240, adaptation_pct=4.869774)])
    monkeypatch.setattr(experiment, 'check_training_memory', lambda cfg, arm: None)

    class BeforeTraining(Exception):
        pass

    def save_arm_config(cfg, split, seed_root, name, resolution, probability):
        save_config(cfg, output / 'resolved.yaml')
        raise BeforeTraining

    monkeypatch.setattr(experiment, 'run_arm', save_arm_config)
    with pytest.raises(BeforeTraining):
        experiment.main()
    hardware = yaml.safe_load((output / 'resolved.yaml').read_text())['comparison_hardware']
    assert type(hardware['torch']) is str
    assert hardware['torch'] == '2.11.0+cu130'
    assert json.loads(json.dumps(hardware)) == hardware


def test_memory_probe_checks_dora_before_any_timed_arm(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import src.qq_lora_dora_ablation as experiment

    monkeypatch.setattr(sys, 'argv', ['comparison', '--memory-check-only'])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(
        name='NVIDIA A100-SXM4-40GB', total_memory=40 * 1024**3))
    arms = [dict(arm='dora_r23', rank=23, dora=True, adaptation_parameters=4200192,
                 original_backbone_parameters=86250240, adaptation_pct=4.869774)]
    monkeypatch.setattr(experiment, 'plan_arms', lambda cfg: arms)
    seen = []
    def probe(cfg, arm):
        seen.append((cfg.batch_size, cfg.grad_accum_steps, arm['arm']))
    monkeypatch.setattr(experiment, 'check_training_memory', probe)
    monkeypatch.setattr(experiment, 'run_arm', lambda *args: pytest.fail('Timed training started'))
    experiment.main()
    assert seen == [(64, 4, 'dora_r23')]


def test_memory_probe_runs_accumulated_updates_and_handles_oom(monkeypatch):
    from contextlib import nullcontext
    import src.qq_lora_dora_ablation as experiment
    from src.heads import CosineClassifier

    cfg = load_config('configs/qq_lora_dora_ablation.yaml')
    cfg.update(dict(batch_size=2, grad_accum_steps=4, img_size=8, num_classes=10))
    arm = dict(arm='dora_r23', rank=23, dora=True)
    calls = []

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature = LoRALinear(torch.nn.Linear(3, 4), 1, 1, dora=True)
            self.head = CosineClassifier(4, 10)
        def to(self, device): return self  # Execute the probe's training path on CPU.
        def get_classifier(self): return self.head
        def forward(self, x):
            calls.append(x.shape[0])
            return self.head(self.feature(x.mean((2, 3))))

    model = Tiny()
    before = model.feature.magnitude.detach().clone()
    def build(probe_cfg):
        assert not probe_cfg.pretrained and probe_cfg.head_init == 'random'
        assert probe_cfg.lora_dora and probe_cfg.lora_rank == 23
        return model
    monkeypatch.setattr(experiment, 'build_model', build)
    monkeypatch.setattr(experiment, 'amp_context', lambda *args: nullcontext())
    randn, arange = torch.randn, torch.arange
    monkeypatch.setattr(torch, 'randn', lambda *args, **kwargs: randn(*args))
    monkeypatch.setattr(torch, 'arange', lambda *args, **kwargs: arange(*args))
    for name in ['empty_cache', 'reset_peak_memory_stats', 'synchronize']:
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 1024**3)
    assert experiment.check_training_memory(cfg, arm) == 1.
    assert calls == [2] * 8  # Two optimizer updates, each with four microbatches.
    assert not torch.equal(before, model.feature.magnitude)
    assert cfg.pretrained and cfg.head_init == 'prototype'

    def oom(probe_cfg):
        raise torch.cuda.OutOfMemoryError('synthetic OOM')
    monkeypatch.setattr(experiment, 'build_model', oom)
    with pytest.raises(RuntimeError, match='before any timed arm started'):
        experiment.check_training_memory(cfg, arm)


def test_dora_starts_at_base_trains_and_merges():
    torch.manual_seed(42)
    base = torch.nn.Linear(7, 9)
    x = torch.randn(3, 4, 7)
    expected = base(x).detach()
    layer = LoRALinear(base, rank=3, alpha=3, dora=True)
    torch.testing.assert_close(layer(x), expected)
    optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        layer(x).square().mean().backward()
        assert layer.magnitude.grad is not None and torch.isfinite(layer.magnitude.grad).all()
        assert layer.lora_B.weight.grad.abs().sum() > 0
        assert base.weight.grad is None
        optimizer.step()
    assert layer.lora_A.weight.grad.abs().sum() > 0
    layer.eval()
    assert not torch.allclose(layer(x), expected)
    torch.testing.assert_close(layer(x), layer.merged()(x), rtol=1e-5, atol=1e-6)


def test_real_vit_preflight_selects_shared_rank_and_matches_runtime():
    torch.set_num_threads(1)
    cfg = load_config('configs/qq_lora_dora_ablation.yaml')
    arms = plan_arms(cfg)
    assert [(a['rank'], a['dora']) for a in arms] == [(24, False), (23, False), (23, True)]
    assert all(a['adaptation_pct'] < 5 for a in arms)
    assert cfg.pretrained and cfg.lora_rank == 24  # Preflight leaves training settings intact.
    for arm in arms:
        model = build_model(Config({**cfg.to_dict(), 'pretrained': False,
                                    'lora_rank': arm['rank'], 'lora_alpha': arm['rank'],
                                    'lora_dora': arm['dora']}))
        assert model.adaptation_stats['adaptation_parameters'] == arm['adaptation_parameters']
        assert model.adaptation_stats['adaptation_pct'] == arm['adaptation_pct']
        del model
    with pytest.raises(ValueError, match='strictly below'):
        build_model(Config({**cfg.to_dict(), 'pretrained': False, 'lora_dora': True}))


def test_common_epochs_exclude_partial_and_second_seed_is_required(tmp_path):
    results = []
    for seed in [42, 43]:
        for dora in [False, True]:
            directory = tmp_path / f'{seed}_{dora}'
            directory.mkdir()
            with (directory / 'epoch_log.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['epoch', 'epoch_complete', 'val_acc', 'val_loss'])
                writer.writeheader()
                writer.writerows([dict(epoch=1, epoch_complete=1, val_acc=.9 + .01 * dora, val_loss=.5),
                                  dict(epoch=2, epoch_complete=int(not dora), val_acc=.95, val_loss=.4)])
            results.append(dict(seed=seed, rank=23, dora=dora, run_dir=str(directory),
                                final_val_acc=.9 + .01 * dora, images_per_second=100 - 20 * dora))
    common = matched_epoch_results(results[:2])
    assert [row['epoch'] for row in common] == [1]
    assert common[0]['delta_pp'] == pytest.approx(1.)
    save_comparison(tmp_path, results[:2], .003)
    assert not json.loads((tmp_path / 'comparison.json').read_text())['promising_across_seeds']
    save_comparison(tmp_path, results, .003)
    comparison = json.loads((tmp_path / 'comparison.json').read_text())
    assert comparison['promising_across_seeds']
    assert comparison['comparisons'][0]['speed_ratio'] == .8


def test_primary_comparison_uses_reference_lora_rank_under_cap(tmp_path):
    results = []
    for rank, dora, accuracy in [(24, False, .9120), (23, False, .9086), (23, True, .9100)]:
        directory = tmp_path / f'{rank}_{dora}'
        directory.mkdir()
        (directory / 'epoch_log.csv').write_text(
            f'epoch,epoch_complete,val_acc,val_loss\n1,1,{accuracy},1.5\n')
        results.append(dict(seed=42, rank=rank, dora=dora, run_dir=str(directory),
                            final_val_acc=accuracy, images_per_second=100))
    save_comparison(tmp_path, results, .003)
    comparison = json.loads((tmp_path / 'comparison.json').read_text())['comparisons'][0]
    assert (comparison['lora_rank'], comparison['dora_rank']) == (24, 23)
    assert comparison['delta_pp'] == pytest.approx(-.20)
    assert comparison['matched_complete_epochs'][0]['lora_val_acc'] == .9120
    assert comparison['shared_rank_control']['delta_pp'] == pytest.approx(.14)


def test_prototype_seed_is_independent_of_training_seed(tmp_path, monkeypatch):
    import src.qq_lora_ablation as ablation
    from PIL import Image
    source = tmp_path / 'data'
    (source / 'images').mkdir(parents=True)
    for index in range(4):
        Image.new('RGB', (32, 32), (index * 60, 40, 90)).save(source / f'images/{index}.png')
    (source / 'train.txt').write_text(''.join(f'images/{i}.png 0\n' for i in range(4)))
    cfg = load_config('configs/qq_lora_dora_ablation.yaml')
    cfg.update(dict(num_classes=3, ablation_classes=1, ablation_holdout=1, batch_size=2,
                    num_workers=0, bf16=False))
    split = ablation.prepare_split(source, tmp_path / 'split', cfg)

    class Features(torch.nn.Module):
        num_features = 3
        def forward_features(self, x): return x.mean((2, 3))
        def forward_head(self, x, pre_logits=False): return x

    monkeypatch.setattr(ablation, 'build_model', lambda cfg: Features())
    output = tmp_path / 'results'
    output.mkdir()
    first = ablation.ensure_prototypes(cfg, split, output)
    cfg.seed = 43
    assert ablation.ensure_prototypes(cfg, split, output) == first
    assert json.loads(first.with_suffix('.json').read_text())['seed'] == 42
    cfg.prototype_seed = 43
    with pytest.raises(ValueError, match='metadata mismatch'):
        ablation.ensure_prototypes(cfg, split, output)


def test_launcher_cells_compile():
    notebook = json.loads(Path('notebooks/colab_launcher_qq_lora_dora.ipynb').read_text())
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell_{index}', 'exec')


def test_real_trainer_saves_dora_state_and_launcher_displays_results(tmp_path, monkeypatch):
    import src.train as trainer
    import src.qq_lora_ablation as ablation
    from src.heads import CosineClassifier
    from PIL import Image
    torch.set_num_threads(1)
    source = tmp_path / 'data'
    (source / 'images').mkdir(parents=True)
    for index in range(8):
        Image.new('RGB', (32, 32), (index * 30, 40, 90)).save(source / f'images/{index}.png')
    (source / 'train.txt').write_text(''.join(f'images/{i}.png {i % 2}\n' for i in range(8)))
    cfg = load_config('configs/qq_lora_dora_ablation.yaml')
    cfg.update(dict(num_classes=10, ablation_classes=2, ablation_holdout=1, batch_size=2,
                    grad_accum_steps=1, num_workers=0, bf16=False, lora_rank=1,
                    lora_alpha=1, max_train_minutes=.002))
    split = ablation.prepare_split(source, tmp_path / 'split', cfg)

    class Tiny(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.feature = LoRALinear(torch.nn.Linear(3, 4), 1, 1, dora=cfg.lora_dora)
            self.head = CosineClassifier(4, 10)
            count = 7 + 4 * cfg.lora_dora
            self.adaptation_stats = dict(adaptation_parameters=count,
                                         original_backbone_parameters=1000,
                                         adaptation_pct=count / 10)
        def get_classifier(self): return self.head
        def forward(self, x): return self.head(self.feature(x.mean((2, 3))))

    monkeypatch.setattr(trainer, 'build_model', Tiny)

    def prototypes(cfg, split, root):
        path = Path(root) / f'prototypes_fit_{cfg.img_size}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(prototypes=torch.ones(10, 4)), path)
        return path

    monkeypatch.setattr(ablation, 'ensure_prototypes', prototypes)
    root = tmp_path / 'results'
    results = []
    for dora in [False, True]:
        cfg.lora_dora = dora
        name = 'dora_r1' if dora else 'lora_r1'
        result = ablation.run_arm(cfg, split, root, name, 32, .25)
        results.append({**result, 'seed': 42, 'rank': 1, 'dora': dora})
        checkpoint = torch.load(Path(result['run_dir']) / 'last.pt', weights_only=False)
        assert ('feature.magnitude' in checkpoint['state_dict']) == dora
        assert 'feature.linear.weight' not in checkpoint['state_dict']
        assert result == ablation.run_arm(cfg, split, root, name, 32, .25)
    ablation.save_results(root, results)
    save_comparison(root, results, .003)
    import matplotlib
    matplotlib.use('Agg')
    notebook = json.loads(Path('notebooks/colab_launcher_qq_lora_dora.ipynb').read_text())
    summary = next(''.join(c['source']) for c in notebook['cells']
                   if c['cell_type'] == 'code' and 'labels = results' in ''.join(c['source']))
    exec(summary, {'EXPERIMENT_DIR': root, 'display': lambda value: None})
    assert (root / 'comparison.png').is_file()
