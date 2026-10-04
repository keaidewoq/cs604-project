from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cs604.config import PATH_DEFAULTS, expand_paths, load_config  # noqa: E402


class ConfigPathTest(unittest.TestCase):
    def test_default_is_sibling_folder(self):
        saved = os.environ.pop("CS604_DATA", None)
        try:
            value = expand_paths("${CS604_DATA}/train.txt")
        finally:
            if saved is not None:
                os.environ["CS604_DATA"] = saved
        self.assertEqual(value, PATH_DEFAULTS["CS604_DATA"] + "/train.txt")

    def test_environment_overrides_default_in_nested_values(self):
        os.environ["CS604_OUTPUTS"] = "/content/drive/MyDrive/cs604/outputs"
        try:
            cfg = expand_paths({"views": {"holdout": ["${CS604_OUTPUTS}/run/splits"]}, "lr": 1e-4})
        finally:
            del os.environ["CS604_OUTPUTS"]
        self.assertEqual(cfg["views"]["holdout"], ["/content/drive/MyDrive/cs604/outputs/run/splits"])
        self.assertEqual(cfg["lr"], 1e-4)

    def test_unknown_variable_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text('output_dir: "${CS604_NOPE}/x"\n', encoding="utf-8")
            with self.assertRaises(SystemExit):
                load_config(path)

    def test_every_repo_config_expands(self):
        for path in sorted((Path(__file__).resolve().parents[1] / "configs").rglob("*.yaml")):
            self.assertNotIn("${", repr(load_config(path)), path.name)


if __name__ == "__main__":
    unittest.main()
