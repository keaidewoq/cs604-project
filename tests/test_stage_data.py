from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_stage_data():
    path = Path(__file__).resolve().parents[1] / "scripts" / "stage_data.py"
    spec = importlib.util.spec_from_file_location("stage_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


stage_data = _load_stage_data()


class StageDataLayoutTests(unittest.TestCase):
    def test_archive_layout_requires_train_txt_and_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(stage_data.archive_layout_ok(root, "train.tar.gz"))
            (root / "train.txt").write_text("a\n", encoding="utf-8")
            self.assertFalse(stage_data.archive_layout_ok(root, "train.tar.gz"))
            (root / "train").mkdir()
            self.assertFalse(stage_data.archive_layout_ok(root, "train.tar.gz"))
            (root / "train.txt").write_text("line\n" * stage_data.TRAIN_TXT_MIN_LINES, encoding="utf-8")
            self.assertTrue(stage_data.archive_layout_ok(root, "train.tar.gz"))

    def test_stale_marker_detected_when_layout_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = stage_data.extract_marker(root, "train.tar.gz")
            marker.write_text("ok\n", encoding="utf-8")
            self.assertFalse(stage_data.archive_layout_ok(root, "train.tar.gz"))


if __name__ == "__main__":
    unittest.main()
