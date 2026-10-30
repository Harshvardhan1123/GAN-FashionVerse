"""Deep convolutional GAN discriminator for 28x28 Fashion-MNIST images."""

from __future__ import annotations

from torch import Tensor, nn


class DCGANDiscriminator(nn.Module):
    """Return one real/fake logit for each 28x28 grayscale image."""

    def __init__(self, feature_maps: int = 64) -> None:
        super().__init__()
        if feature_maps < 1:
            raise ValueError("feature_maps must be greater than zero")
        self.network = nn.Sequential(
            nn.Conv2d(1, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(feature_maps * 4 * 4 * 4, 1),
        )
        self.apply(_initialize_weights)

    def forward(self, images: Tensor) -> Tensor:
        """Return one unnormalized real/fake logit per image."""
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError(
                "images must have shape (batch, 1, 28, 28), "
                f"received {tuple(images.shape)}"
            )
        return self.network(images)


def _initialize_weights(module: nn.Module) -> None:
    """Apply the original DCGAN normal initialization convention."""
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
