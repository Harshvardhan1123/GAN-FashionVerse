"""Basic fully connected GAN discriminator for Fashion-MNIST."""

from __future__ import annotations

from torch import Tensor, nn


class BasicDiscriminator(nn.Module):
    """Classify flattened 28x28 grayscale images using output logits."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return one unnormalized real/fake logit per image."""
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError(
                "images must have shape (batch, 1, 28, 28), "
                f"received {tuple(images.shape)}"
            )
        return self.network(images)
