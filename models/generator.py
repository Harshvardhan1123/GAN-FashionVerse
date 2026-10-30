"""Basic fully connected GAN generator for Fashion-MNIST."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BasicGenerator(nn.Module):
    """Map latent noise vectors to normalized 28x28 grayscale images."""

    def __init__(self, latent_dim: int = 100) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be greater than zero")
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, noise: Tensor) -> Tensor:
        """Generate images from noise shaped ``(batch, latent_dim)``."""
        if noise.ndim != 2 or noise.shape[1] != self.latent_dim:
            raise ValueError(
                f"noise must have shape (batch, {self.latent_dim}), "
                f"received {tuple(noise.shape)}"
            )
        return self.network(noise).view(-1, 1, 28, 28)


def sample_noise(
    batch_size: int,
    latent_dim: int = 100,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Sample standard normal latent vectors for generator inference/training."""
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")
    if latent_dim < 1:
        raise ValueError("latent_dim must be greater than zero")
    return torch.randn(batch_size, latent_dim, device=device)
