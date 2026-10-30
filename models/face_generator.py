"""Independent 64x64 RGB GAN architecture for a future face dataset."""

from __future__ import annotations

from torch import Tensor, nn


class FaceGenerator(nn.Module):
    """Generate 64x64 RGB images from latent vectors."""

    def __init__(self, latent_dim: int = 100, feature_maps: int = 64) -> None:
        super().__init__()
        if latent_dim < 1 or feature_maps < 1:
            raise ValueError("latent_dim and feature_maps must be greater than zero")
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(_initialize_weights)

    def forward(self, noise: Tensor) -> Tensor:
        """Generate RGB images from ``(batch, latent_dim)`` noise."""
        if noise.ndim != 2 or noise.shape[1] != self.latent_dim:
            raise ValueError(
                f"noise must have shape (batch, {self.latent_dim}), "
                f"received {tuple(noise.shape)}"
            )
        return self.network(noise[:, :, None, None])


class FaceDiscriminator(nn.Module):
    """Classify 64x64 RGB images using a single real/fake logit."""

    def __init__(self, feature_maps: int = 64) -> None:
        super().__init__()
        if feature_maps < 1:
            raise ValueError("feature_maps must be greater than zero")
        self.network = nn.Sequential(
            nn.Conv2d(3, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
        )
        self.apply(_initialize_weights)

    def forward(self, images: Tensor) -> Tensor:
        """Return one unnormalized real/fake logit per RGB image."""
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 64, 64):
            raise ValueError(
                "images must have shape (batch, 3, 64, 64), "
                f"received {tuple(images.shape)}"
            )
        return self.network(images)


def _initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
