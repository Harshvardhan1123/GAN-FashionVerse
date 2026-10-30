"""Image transforms shared by Fashion-MNIST training and inspection."""

from __future__ import annotations

from torchvision import transforms


def fashion_mnist_transform() -> transforms.Compose:
    """Return the canonical transform for the GAN's image range."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
