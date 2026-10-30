"""Real-face dataset loading for the independent face GAN."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class FaceImageDataset(Dataset[tuple[Tensor, int]]):
    """Load RGB face images recursively from a directory."""

    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root: str | Path, image_size: int = 64) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Face dataset directory does not exist: {self.root}")
        if image_size < 1:
            raise ValueError("image_size must be greater than zero")
        self.paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in self.extensions
        )
        if not self.paths:
            raise ValueError(f"No supported face images found in {self.root}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), 0


def create_face_loader(
    root: str | Path,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[FaceImageDataset, "DataLoader[tuple[Tensor, int]]"]:
    """Create a Windows-safe DataLoader for real face images."""
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    from torch.utils.data import DataLoader

    dataset = FaceImageDataset(root)
    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
