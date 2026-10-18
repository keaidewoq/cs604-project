"""Checks for crop geometry, fitting-only splits and equal-time arm execution."""
import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import RandomCropOrPad, ResizeWithPadding, build_transforms
from src.qq_lora_ablation import candidate_384, partition_rows, prepare_split, run_arm
from src.utils import Config, load_config, set_seed


def test_padding_preserves_full_image_and_aspect():
    image = Image.new('RGB', (500, 375), (255, 0, 0))
    padded = np.asarray(ResizeWithPadding(336, (128,128,128))(image))
    assert padded.shape == (336,336,3)
    assert np.all(padded[:42] == 128) and np.all(padded[-42:] == 128)
    assert np.all(padded[42:294] == (255,0,0))
    for size in [(1,999),(999,1),(333,100),(100,333)]:
        assert ResizeWithPadding(384, (128,128,128))(Image.new('RGB',size)).size == (384,384)


def test_each_visit_uses_exactly_one_branch():
    calls = []
    def crop(image): calls.append('crop'); return image
    def pad(image): calls.append('pad'); return image
    tf = RandomCropOrPad(crop,pad,.25)
    random.seed(42)
    for _ in range(10000): tf(None)
    assert len(calls) == 10000
    assert .23 < calls.count('pad')/len(calls) < .27
    for probability,expected in [(0,'crop'),(1,'pad')]:
        calls.clear()
        RandomCropOrPad(crop,pad,probability)(None)
        assert calls == [expected]


def test_baseline_unchanged_and_mixed_retains_other_transforms():
    cfg = load_config('configs/qq_lora_ablation.yaml')
    image = Image.new('RGB',(500,375),(30,150,200))
    a,_ = build_transforms(cfg)
    cfg.full_image_probability = .25
    b,_ = build_transforms(cfg)
    assert repr(a.transforms[1:]) == repr(b.transforms[1:])
    assert isinstance(b.transforms[0], RandomCropOrPad)
    cfg.full_image_probability = 0
    baseline,_ = build_transforms(cfg)
    set_seed(42); first = a(image)
    set_seed(42); second = baseline(image)
    assert torch.equal(first,second)
    cfg.full_image_probability = 1
    padded,eval_tf = build_transforms(cfg)
    assert padded(image).shape == (3,336,336)
    assert eval_tf(image).shape == (3,336,336)
    cfg.full_image_probability = float('nan')
    with pytest.raises(ValueError): build_transforms(cfg)


def test_partition_is_reproducible_and_keeps_original_labels():
    rows = [(f'images/{label}_{i}.png',label) for label in [2,300,9999] for i in range(9)]
    fit,val = partition_rows(rows,2,5,42)
    assert (fit,val) == partition_rows(rows,2,5,42)
    assert not {p for p,_ in fit} & {p for p,_ in val}
    assert len(fit)==8 and len(val)==10
    assert {y for _,y in fit} <= {2,300,9999}


def test_saved_split_detects_changed_source(tmp_path):
    source=tmp_path/'data'; source.mkdir()
    (source/'images').mkdir()
    text=''.join(f'images/{c}_{i}.png {c}\n' for c in range(3) for i in range(8))
    (source/'train.txt').write_text(text)
    cfg=Config(dict(ablation_classes=2,ablation_holdout=2,seed=42,num_classes=10000))
    split=prepare_split(source,tmp_path/'experiment',cfg)
    assert (split/'images').resolve() == (source/'images').resolve()
    assert prepare_split(source,tmp_path/'experiment',cfg)==split
    (source/'train.txt').write_text(text+'images/extra.png 0\n')
    with pytest.raises(ValueError,match='differs'):
        prepare_split(source,tmp_path/'experiment',cfg)


def test_optional_gate_uses_best_336_and_preserves_crop_choice():
    rows=[dict(arm=a,resolution=r,full_image_probability=p,final_val_acc=score)
          for a,r,p,score in [('A',336,0,.90),('B',336,.25,.91),('C',384,0,.905),('D',384,.25,.909)]]
    assert candidate_384(rows,.003) is None
    rows[-1]['final_val_acc']=.92
    assert candidate_384(rows,.003) == rows[-1]


def test_arm_runs_real_trainer_on_tiny_images_and_reuses_result(tmp_path,monkeypatch):
    import src.train as trainer
    import src.qq_lora_ablation as ablation
    from src.heads import CosineClassifier
    torch.set_num_threads(1)
    source=tmp_path/'data'; source.mkdir()
    image_dir=source/'images'; image_dir.mkdir()
    rows=[]
    for c in range(2):
        for i in range(8):
            name=f'images/{c}_{i}.png'
            Image.new('RGB',(36,28),(220,40,20) if c else (20,40,220)).save(source/name)
            rows.append(f'{name} {c}\n')
    (source/'train.txt').write_text(''.join(rows))
    cfg=load_config('configs/qq_lora_ablation.yaml')
    cfg.update(dict(num_classes=10,ablation_classes=2,ablation_holdout=2,num_workers=0,
                    batch_size=4,grad_accum_steps=1,max_train_minutes=.002,bf16=False))
    split=prepare_split(source,tmp_path/'local',cfg)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head=CosineClassifier(3,10)
            self.adaptation_stats=dict(adaptation_parameters=3,original_backbone_parameters=100,
                                       adaptation_pct=3.)
        def get_classifier(self):return self.head
        def forward(self,x):return self.head(x.mean((2,3)))
    monkeypatch.setattr(trainer,'build_model',lambda cfg:Tiny())
    built=[]
    def prototypes(cfg,split,root):
        # Smoke fixture: no model download; all real loader/trainer/logging paths run.
        path=Path(root)/f'prototypes_fit_{cfg.img_size}.pt';path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(prototypes=torch.randn(10,3)),path)
        built.append(str(split))
        return path
    monkeypatch.setattr(ablation,'ensure_prototypes',prototypes)
    experiment=tmp_path/'runs'
    result=run_arm(cfg,split,experiment,'A',336,0)
    assert result['training_minutes'] >= cfg.max_train_minutes
    assert result['images_per_second'] > 0 and result['epochs'] > 0
    assert (Path(result['run_dir'])/'last.pt').exists()
    assert 0 <= result['final_val_acc'] <= 1
    again=run_arm(cfg,split,experiment,'A',336,0)
    assert again==result and len(built)==1
    results=[result]
    for name,resolution,probability in ablation.ARMS[1:]:
        results.append(run_arm(cfg,split,experiment,name,resolution,probability))
    ablation.save_results(experiment,results)
    assert len(json.loads((experiment/'results.json').read_text()))==4
    assert all(r['training_minutes'] >= cfg.max_train_minutes for r in results)
    write_selection=dict(candidate_384=ablation.candidate_384(results,.003),min_gain=.003)
    (experiment/'selection.json').write_text(json.dumps(write_selection))
    # Execute the notebook's actual summary/plot cell against trainer outputs.
    import matplotlib
    matplotlib.use('Agg')
    notebook=json.loads(Path('notebooks/colab_launcher_qq.ipynb').read_text())
    summary_cell=next(c['source'] for c in notebook['cells'] if c['cell_type']=='code' and 'main_results =' in c['source'])
    exec(summary_cell, {'EXPERIMENT_DIR':experiment,'display':lambda value:None})
    assert (experiment/'comparison.png').is_file()
    with pytest.raises(ValueError,match='differs'):
        run_arm(cfg,split,experiment,'A',336,.25)


def test_prototypes_use_only_fitting_images_and_cache_by_resolution(tmp_path,monkeypatch):
    import src.qq_lora_ablation as ablation
    source=tmp_path/'data';source.mkdir();(source/'images').mkdir()
    for c in range(2):
        for i in range(8):
            Image.new('RGB',(40,30),(240,80,20) if c else (20,80,240)).save(source/f'images/{c}_{i}.png')
    (source/'train.txt').write_text(''.join(f'images/{c}_{i}.png {c}\n' for c in range(2) for i in range(8)))
    cfg=load_config('configs/qq_lora_ablation.yaml')
    cfg.update(dict(num_classes=10,ablation_classes=2,ablation_holdout=2,num_workers=0,batch_size=4,bf16=False))
    split=prepare_split(source,tmp_path/'local',cfg)
    class Features(torch.nn.Module):
        num_features=3
        def forward_features(self,x):return x.mean((2,3))
        def forward_head(self,x,pre_logits=False):return x
    monkeypatch.setattr(ablation,'build_model',lambda cfg:Features())
    output=tmp_path/'output';output.mkdir()
    p=ablation.ensure_prototypes(cfg,split,output)
    blob=torch.load(p,weights_only=True)
    assert blob['samples_per_class'].tolist()==[6,6]+[0]*8
    assert torch.allclose(blob['prototypes'].norm(dim=1),torch.ones(10))
    assert torch.isfinite(blob['prototypes']).all()
    assert blob['fit_sha256']==ablation.file_hash(split/'train.txt')
    # Holdout colors cannot affect prototypes: they are never loaded.
    for line in (split/'val.txt').read_text().splitlines():
        Image.new('RGB',(40,30),(0,255,0)).save(source/line.split()[0])
    assert ablation.ensure_prototypes(cfg,split,output)==p
    cfg.img_size=384
    other=ablation.ensure_prototypes(cfg,split,output)
    assert other!=p and torch.load(other,weights_only=True)['img_size']==384
    cfg.eval_crop_pct=.75
    with pytest.raises(ValueError,match='metadata mismatch'):
        ablation.ensure_prototypes(cfg,split,output)
