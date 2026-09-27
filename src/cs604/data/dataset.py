"""Image-list dataset for the CS604 course splits.

train.txt lines: ``train/class_XXXX/<id>.jpg <class_id>``
val.txt / test.txt lines: ``val/image_XXXXXX.jpg`` (path only; labels withheld)
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_split_index(list_path: Path) -> list[tuple[str, int | None]]:
    samples: list[tuple[str, int | None]] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            rel_path = parts[0]
            label = int(parts[1]) if len(parts) > 1 else None
            samples.append((rel_path, label))
    return samples


class CS604ImageDataset:
    """Map-style dataset. Returns ``(image, label, relative_path)``.

    ``label`` is ``None`` for the unlabeled validation and test splits.
    """

    def __init__(self, root: str | Path, split: str, transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        list_path = self.root / f"{split}.txt"
        if not list_path.is_file():
            raise FileNotFoundError(f"Missing split index: {list_path}")
        self.samples = load_split_index(list_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, label = self.samples[index]
        image = Image.open(self.root / rel_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, rel_path
