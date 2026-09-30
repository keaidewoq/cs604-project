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
