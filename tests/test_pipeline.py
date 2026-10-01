from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cs604.data.dataset import stratified_split
from cs604.engine.predict import validate_submission
from cs604.models.build import CosineClassifier, LoRALinear, apply_class_prototypes


class SplitTests(unittest.TestCase):
    def test_stratified_split_is_balanced_and_repeatable(self) -> None:
        samples = [
            (f"train/class_{label}/{index}.jpg", label)
            for label in range(3)
            for index in range(8)
        ]
        first = stratified_split(samples, val_per_class=2, train_per_class=3, seed=7)
        second = stratified_split(samples, val_per_class=2, train_per_class=3, seed=7)
        self.assertEqual(first, second)
        train, val = first
        self.assertEqual(len(train), 9)
        self.assertEqual(len(val), 6)
        self.assertEqual({label for _path, label in train}, {0, 1, 2})


class LoRATests(unittest.TestCase):
    def test_only_lora_weights_are_trainable_and_reloadable(self) -> None:
        layer = LoRALinear(nn.Linear(5, 7), rank=2, alpha=4)
        self.assertFalse(layer.base.weight.requires_grad)
        self.assertTrue(layer.lora_a.requires_grad)
        inputs = torch.randn(3, 5)
        original = layer(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(layer.state_dict(), path)
            reloaded = LoRALinear(nn.Linear(5, 7), rank=2, alpha=4)
            reloaded.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(original, reloaded(inputs))
            for key, value in layer.state_dict().items():
                torch.testing.assert_close(value, reloaded.state_dict()[key])


class TinyViT(nn.Module):
    """Just enough of a timm ViT for build_vit: blocks[].attn.qkv, norm, head."""

    def __init__(self, seed: int, width: int = 4, classes: int = 3):
        super().__init__()
        torch.manual_seed(seed)
        self.patch = nn.Linear(width, width)
        self.blocks = nn.ModuleList(
            nn.Module() for _ in range(2)
        )
        for block in self.blocks:
            block.attn = nn.Module()
            block.attn.qkv = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, classes)

    def get_classifier(self):
        return self.head

    def set_grad_checkpointing(self, enabled: bool) -> None:
        pass


class CompactCheckpointTests(unittest.TestCase):
    CFG = {"timm_checkpoint": "tiny", "num_classes": 3, "mode": "lora", "lora_rank": 2,
           "head_type": "cosine", "train_head": False}

    def _build(self, seed: int):
        from unittest.mock import patch

        from cs604.models.build import build_from_config

        with patch("cs604.models.build.timm.create_model", return_value=TinyViT(seed)):
            return build_from_config(self.CFG)

    def test_compact_keeps_only_trained_and_head_weights_and_round_trips(self) -> None:
        from cs604.models.build import checkpoint_model_state, load_model_state

        model = self._build(seed=0)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora" in name or name.startswith("head."):
                    parameter.add_(1.0)
        saved = checkpoint_model_state(model, compact=True)
        self.assertEqual(saved["model_format"], "compact")
        self.assertTrue(all("lora" in key or key.startswith("head.") for key in saved["model"]))
        self.assertIn("head.weight", saved["model"])  # frozen but prototype-initialized
        self.assertNotIn("patch.weight", saved["model"])

        restored = self._build(seed=0)
        load_model_state(restored, saved)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)

        with self.assertRaisesRegex(ValueError, "Frozen backbone differs"):
            load_model_state(self._build(seed=1), saved)

    def test_full_checkpoints_still_load(self) -> None:
        from cs604.models.build import checkpoint_model_state, load_model_state

        model = self._build(seed=0)
        restored = self._build(seed=1)
        load_model_state(restored, {"model": model.state_dict()})  # pre-compact checkpoints
        torch.testing.assert_close(restored.patch.weight, model.patch.weight)
        self.assertEqual(checkpoint_model_state(model, compact=False)["model_format"], "full")

    def test_compact_default_only_for_frozen_pretrained_backbone(self) -> None:
        from cs604.models.build import use_compact_checkpoints

        self.assertTrue(use_compact_checkpoints({"mode": "lora"}))
        self.assertTrue(use_compact_checkpoints({"mode": "head"}))
        self.assertFalse(use_compact_checkpoints({"mode": "full"}))
        self.assertFalse(use_compact_checkpoints({"mode": "lora", "init_checkpoint": "x.pt"}))
        self.assertFalse(use_compact_checkpoints({"mode": "lora", "compact_checkpoints": False}))
        with self.assertRaisesRegex(ValueError, "compact_checkpoints"):
            use_compact_checkpoints({"mode": "full", "compact_checkpoints": True})


class PrototypeTests(unittest.TestCase):
    def test_prototypes_replace_classifier_weights(self) -> None:
        layer = CosineClassifier(4, 3)
        prototypes = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]]
        )
        apply_class_prototypes(type("Model", (), {"get_classifier": lambda self: layer})(), prototypes)
        torch.testing.assert_close(layer.weight[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(layer.weight[1], torch.tensor([0.0, 1.0, 0.0, 0.0]))


class SubmissionTests(unittest.TestCase):
    def test_submission_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "val.txt"
            submission = root / "val_team_01.txt"
            index.write_text("val/a.jpg\nval/b.jpg\n", encoding="utf-8")
            submission.write_text("val/a.jpg 0\nval/b.jpg 9999\n", encoding="utf-8")
            validate_submission(submission, index)
            submission.write_text("val/b.jpg 0\nval/a.jpg 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected"):
                validate_submission(submission, index)


if __name__ == "__main__":
    unittest.main()
