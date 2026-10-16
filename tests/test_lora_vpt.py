"""CPU checks for prompt semantics, PEFT checkpoints and ablation provenance."""

from argparse import Namespace

import pytest
import torch
from PIL import Image
from timm.models.vision_transformer import VisionTransformer

from src.heads import replace_head
from src.lora_vpt_ablation import (
    choose_prompt_lr, prepare_split, run_one,
    write_comparison,
)
from src.model import apply_method
from src.peft.lora import merge_lora
from src.predict import load_checkpoint
from src.train import adamw_param_groups, save_checkpoint
from src.utils import Config, set_seed


def tiny_config(mode="none", length=0):
    return Config({"method": "lora", "pretrained": False, "num_classes": 3,
                   "model_name": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                   "img_size": 16, "head_type": "cosine", "head_init": "random",
                   "lora_targets": ["qkv", "proj", "fc1", "fc2"],
                   "lora_rank": 2, "lora_alpha": 2, "lora_dropout": 0.0,
                   "unfreeze_extra": ["norm", "bias", "pos_embed", "cls_token"],
                   "prompt_mode": mode, "prompt_length": length,
                   "seed": 42, "batch_size": 8, "grad_accum_steps": 2,
                   "num_workers": 0, "epochs": 1, "warmup": 0,
                   "bf16": False, "weight_decay": 0.05, "lr": 3e-4,
                   "head_lr": 3e-3, "prompt_lr": 1e-3, "label_smoothing": 0.1})


def tiny_model(cfg):
    model = VisionTransformer(img_size=16, patch_size=4, embed_dim=24,
                              depth=3, num_heads=3, mlp_ratio=2,
                              num_classes=cfg.num_classes)
    replace_head(model, cfg)
    return apply_method(model, cfg)


@pytest.mark.parametrize("mode", ["shallow", "deep"])
def test_token_placement_and_gradients(mode):
    cfg = tiny_config(mode, 2)
    model = tiny_model(cfg)
    inputs = []
    for block in model.blocks:
        block.register_forward_pre_hook(lambda _module, args: inputs.append(args[0].detach().clone()))
    logits = model(torch.randn(2, 3, 16, 16))
    assert logits.shape == (2, 3)
    assert all(x.shape == (2, 19, 24) for x in inputs)
    assert torch.equal(inputs[0][:, 1:3], model.vpt.embeddings[0].expand(2, -1, -1))
    if mode == "deep":
        for index in range(3):
            assert torch.equal(inputs[index][:, 1:3], model.vpt.embeddings[index].expand(2, -1, -1))
    else:
        assert not torch.equal(inputs[1][:, 1:3], inputs[0][:, 1:3])
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(g.abs().sum() > 0 for g in model.vpt.embeddings.grad)
    assert model.blocks[0].attn.qkv.lora_B.weight.grad.abs().sum() > 0
    assert model.patch_embed.proj.weight.grad is None


def test_baseline_weights_and_rng_unchanged_by_prompt_initialization():
    set_seed(42)
    base = tiny_model(tiny_config())
    next_base = torch.rand(4)
    set_seed(42)
    prompted = tiny_model(tiny_config("deep", 8))
    next_prompted = torch.rand(4)
    assert not hasattr(base, "vpt")
    assert torch.equal(next_base, next_prompted)
    for name, weight in base.state_dict().items():
        assert torch.equal(weight, prompted.state_dict()[name]), name


@pytest.mark.parametrize("mode", ["none", "shallow", "deep"])
def test_optimizer_checkpoint_reload_and_merge(tmp_path, mode):
    cfg = tiny_config(mode, 0 if mode == "none" else 2)
    set_seed(7)
    model = tiny_model(cfg)
    groups = adamw_param_groups(model, .05, 3e-4, head_lr=3e-3, prompt_lr=1e-3)
    assigned = [id(p) for group in groups for p in group["params"]]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {id(p) for p in model.parameters() if p.requires_grad}
    if mode != "none":
        group = next(g for g in groups if any(p is model.vpt.embeddings for p in g["params"]))
        assert group["lr"] == 1e-3 and group["weight_decay"] == 0
    images = torch.randn(2, 3, 16, 16)
    torch.nn.functional.cross_entropy(model(images), torch.tensor([0, 1])).backward()
    torch.optim.AdamW(groups).step()
    model.eval()
    expected = model(images).detach()
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(checkpoint, model, cfg, 1, {"val_acc": .5})
    set_seed(7)
    restored = tiny_model(cfg).eval()
    payload = load_checkpoint(restored, checkpoint, torch.device("cpu"))
    assert payload["trainable_only"]
    assert ("vpt.embeddings" in payload["state_dict"]) == (mode != "none")
    torch.testing.assert_close(restored(images), expected)
    merge_lora(restored)
    torch.testing.assert_close(restored(images), expected, atol=2e-5, rtol=2e-5)


def test_parameter_cap_counts_prompts_and_extras():
    cfg = tiny_config("deep", 8)
    model = tiny_model(cfg)
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad and id(p) not in head_ids)
    assert model.adaptation_stats["adaptation_parameters"] == trainable
    cfg.max_adaptation_pct = model.adaptation_stats["adaptation_pct"]
    with pytest.raises(ValueError, match="strictly below"):
        tiny_model(cfg)


def make_manifest(root, images=False):
    root.mkdir()
    lines = []
    for label in [10, 20, 30]:
        for i in range(45):
            name = f"{label}_{i}.png"
            if images:
                Image.new("RGB", (20, 20), color=(label * 5, i * 5, 0)).save(root / name)
            lines.append(f"{name} {label}\n")
    (root / "train.txt").write_text("".join(lines))


def test_split_is_disjoint_reproducible_and_remapped(tmp_path):
    source = tmp_path / "data"
    make_manifest(source)
    root, info = prepare_split(source, tmp_path / "split", 2, 5, 42)
    fit = [line.rsplit(maxsplit=1) for line in (root / "train.txt").read_text().splitlines()]
    val = [line.rsplit(maxsplit=1) for line in (root / "val.txt").read_text().splitlines()]
    assert len(fit) == 80 and len(val) == 10
    assert not {r[0] for r in fit} & {r[0] for r in val}
    assert {r[1] for r in fit} == {"0", "1"}
    assert prepare_split(source, root, 2, 5, 42)[1] == info
    with pytest.raises(ValueError, match="settings changed"):
        prepare_split(source, root, 2, 5, 43)
    (root / "train.txt").write_text("changed")
    with pytest.raises(ValueError, match="manifest changed"):
        prepare_split(source, root, 2, 5, 42)


def test_lr_tie_rule_and_comparison(tmp_path):
    assert choose_prompt_lr([{"prompt_lr": 3e-4, "best_val_acc": .8},
                             {"prompt_lr": 1e-3, "best_val_acc": .802}]) == 3e-4
    rows = write_comparison([{"arm": "A", "seed": 42, "best_val_acc": .7},
                             {"arm": "B", "seed": 42, "best_val_acc": .72},
                             {"arm": "B", "seed": 43, "best_val_acc": .73}], tmp_path / "comparison.csv")
    assert rows[1]["delta_vs_A_pp"] == pytest.approx(2)
    assert rows[2]["delta_vs_A_pp"] is None


def test_training_runner_completion_and_reuse(tmp_path, monkeypatch):
    import src.train as trainer
    import src.lora_vpt_ablation as runner

    source = tmp_path / "data"
    make_manifest(source, images=True)
    root, _ = prepare_split(source, tmp_path / "split", 2, 5, 42)
    monkeypatch.setattr(trainer, "build_model", tiny_model)
    monkeypatch.setattr(runner, "build_model", tiny_model)
    cfg = tiny_config("deep", 2)
    cfg.num_classes = 2
    cfg.data_root = str(root)
    cfg.prototype_file = str(tmp_path / "prototypes.pt")
    torch.save({"prototypes": torch.randn(2, 24)}, cfg.prototype_file)
    cfg.head_init = "prototype"
    result = run_one(cfg, tmp_path / "run", {"test": True}, "E")
    assert result["best_val_acc"] is not None
    assert result["prompt_parameters"] == 3 * 2 * 24
    monkeypatch.setattr(runner, "train", lambda cfg: pytest.fail("completed run should be skipped"))
    assert run_one(cfg, tmp_path / "run", {"test": True}, "E") == result
    cfg.prompt_lr = .002
    with pytest.raises(ValueError, match="settings changed"):
        run_one(cfg, tmp_path / "run", {"test": True}, "E")


def test_screen_shares_pilot_lr_and_fitting_prototypes(tmp_path, monkeypatch):
    import src.lora_vpt_ablation as runner

    source = tmp_path / "data"
    make_manifest(source)
    calls = []

    def prototypes(cfg, directory):
        fit = (runner.Path(cfg.data_root) / "train.txt").read_text().splitlines()
        assert len(fit) == 80 and cfg.num_classes == 2
        return tmp_path / "fit_only.pt"

    def run(cfg, directory, context, arm):
        calls.append(cfg.to_dict())
        return {"arm": arm, "seed": cfg.seed, "prompt_lr": cfg.prompt_lr,
                "best_val_acc": .8 if cfg.prompt_lr == 3e-4 else .802}

    monkeypatch.setattr(runner, "ensure_prototypes", prototypes)
    monkeypatch.setattr(runner, "parameter_stats", lambda cfg: {})
    monkeypatch.setattr(runner, "run_one", run)
    args = Namespace(data_root=str(source), output_dir=str(tmp_path / "experiment"),
                     config="configs/lora_vpt_ablation.yaml", classes=2, holdout=5,
                     split_seed=42, epochs=5, workers=0, batch_size=8, accum=2,
                     seeds=[42], arms=["A", "B", "E", "C", "D"], lr_pilot=True,
                     prompt_lr=.001, pilot_epochs=3, comparison_stem="comparison")
    results = runner.screen(args)
    assert len(results) == 5 and len(calls) == 7
    assert all(c["epochs"] == 3 for c in calls[:2])
    assert all(c["epochs"] == 5 and c["warmup"] == .25 for c in calls[2:])
    assert all(c["prompt_lr"] == 3e-4 for c in calls[2:])
    assert len({c["prototype_file"] for c in calls}) == 1
    assert all(c["max_train_minutes"] is None for c in calls)


def test_final_rejects_subset_labels(tmp_path):
    from src.lora_vpt_ablation import final_run

    source = tmp_path / "data"
    make_manifest(source)
    args = Namespace(data_root=str(source), output_dir=str(tmp_path / "final"),
                     config="configs/lora_vpt_ablation.yaml", arm="B", seed=42,
                     prompt_lr=.001, minutes=144, workers=0, batch_size=8, accum=2)
    with pytest.raises(ValueError, match="Final training requires original IDs"):
        final_run(args)
