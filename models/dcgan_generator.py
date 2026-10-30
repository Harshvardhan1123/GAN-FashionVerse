"""Deep convolutional GAN generator for 28x28 Fashion-MNIST images."""

from __future__ import annotations

from torch import Tensor, nn


class DCGANGenerator(nn.Module):
    """Generate 28x28 grayscale images from latent vectors."""

    def __init__(self, latent_dim: int = 100, feature_maps: int = 64) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be greater than zero")
        if feature_maps < 1:
            raise ValueError("feature_maps must be greater than zero")
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.Conv2d(feature_maps, 1, 3, 1, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(_initialize_weights)

    def forward(self, noise: Tensor) -> Tensor:
        """Generate images from noise shaped ``(batch, latent_dim)``."""
        if noise.ndim != 2 or noise.shape[1] != self.latent_dim:
            raise ValueError(
                f"noise must have shape (batch, {self.latent_dim}), "
                f"received {tuple(noise.shape)}"
            )
        return self.network(noise.view(noise.shape[0], self.latent_dim, 1, 1))


def _initialize_weights(module: nn.Module) -> None:
    """Apply the original DCGAN normal initialization convention."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
