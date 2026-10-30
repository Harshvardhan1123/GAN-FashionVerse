"""Fashion-MNIST dataset access, metadata, and visualization helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

from data.transforms import fashion_mnist_transform

LOGGER = logging.getLogger(__name__)

FASHION_MNIST_CLASSES: Final[tuple[str, ...]] = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
IMAGE_SIZE: Final[tuple[int, int]] = (28, 28)
CHANNELS: Final[int] = 1


def get_fashion_mnist_dataset(
    root: str | Path = "datasets/fashion_mnist",
    *,
    train: bool = True,
    download: bool = True,
) -> FashionMNIST:
    """Load Fashion-MNIST, downloading it only when requested and missing."""
    dataset_root = Path(root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    return FashionMNIST(
        root=str(dataset_root),
        train=train,
        transform=fashion_mnist_transform(),
        download=download,
    )


def create_fashion_mnist_loader(
    root: str | Path = "datasets/fashion_mnist",
    *,
    batch_size: int = 128,
    train: bool = True,
    download: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Create a Windows-safe DataLoader for Fashion-MNIST."""
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    dataset = get_fashion_mnist_dataset(root, train=train, download=download)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def dataset_metadata(dataset: FashionMNIST) -> dict[str, object]:
    """Return factual metadata for a loaded Fashion-MNIST split."""
    images, labels = dataset.data, dataset.targets
    return {
        "samples": len(dataset),
        "image_dimensions": IMAGE_SIZE,
        "channels": CHANNELS,
        "classes": len(FASHION_MNIST_CLASSES),
        "class_names": FASHION_MNIST_CLASSES,
        "labels_available": len(labels),
        "raw_image_dtype": str(images.dtype),
    }


def denormalize_images(images: Tensor) -> Tensor:
    """Convert normalized GAN tensors in [-1, 1] back to display range [0, 1]."""
    return ((images.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)


def plot_dataset_samples(
    dataset: FashionMNIST,
    *,
    count: int = 16,
    columns: int = 4,
) -> plt.Figure:
    """Build a labeled sample grid without displaying or saving it."""
    if count < 1:
        raise ValueError("count must be greater than zero")
    if columns < 1:
        raise ValueError("columns must be greater than zero")

    sample_count = min(count, len(dataset))
    rows = (sample_count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2))
    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for index in range(sample_count):
        image, label = dataset[index]
        display_image = denormalize_images(image).squeeze(0)
        flat_axes[index].imshow(display_image, cmap="gray", vmin=0.0, vmax=1.0)
        flat_axes[index].set_title(FASHION_MNIST_CLASSES[int(label)])
        flat_axes[index].axis("off")

    for axis in flat_axes[sample_count:]:
        axis.axis("off")

    figure.tight_layout()
    return figure
