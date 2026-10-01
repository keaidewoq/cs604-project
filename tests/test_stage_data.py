from __future__ import annotations

import importlib.util
import subprocess
import sys
import tarfile
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


class PipeIntoTarTests(unittest.TestCase):
    def _make_archive(self, root: Path) -> Path:
        src = root / "src"
        (src / "train" / "0").mkdir(parents=True)
        (src / "train" / "0" / "a.jpg").write_bytes(b"jpeg")
        (src / "train.txt").write_text("train/0/a.jpg 0\n", encoding="utf-8")
        archive = root / "train.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(src / "train", arcname="train")
            handle.add(src / "train.txt", arcname="train.txt")
        return archive

    def test_streams_archive_into_local_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self._make_archive(root)
            out = root / "out"
            # Mimic curl: read config from stdin, write the archive bytes to stdout.
            source = [
                sys.executable,
                "-c",
                "import sys, shutil; sys.stdin.read(); "
                f"shutil.copyfileobj(open({str(archive)!r}, 'rb'), sys.stdout.buffer)",
            ]
            stage_data.pipe_into_tar(source, out, source_input=b"header = x\n")
            self.assertEqual((out / "train" / "0" / "a.jpg").read_bytes(), b"jpeg")
            self.assertTrue((out / "train.txt").is_file())
            self.assertFalse((out / "train.tar.gz").exists())

    def test_failed_download_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = [sys.executable, "-c", "import sys; sys.exit(22)"]
            with self.assertRaises(subprocess.CalledProcessError):
                stage_data.pipe_into_tar(source, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
