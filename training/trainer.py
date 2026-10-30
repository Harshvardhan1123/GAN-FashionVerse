"""Adversarial training loop for the Basic GAN."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn

from models.discriminator import BasicDiscriminator
from models.dcgan_discriminator import DCGANDiscriminator
from models.dcgan_generator import DCGANGenerator
from models.generator import BasicGenerator, sample_noise
from training.utils import save_checkpoint, save_history, save_sample_grid

LOGGER = logging.getLogger(__name__)


class BasicGANTrainer:
    """Train a Basic GAN and persist factual metrics and generated samples."""

    def __init__(
        self,
        generator: BasicGenerator | None = None,
        discriminator: BasicDiscriminator | None = None,
        *,
        latent_dim: int = 100,
        learning_rate: float = 0.0002,
        beta1: float = 0.5,
        device: torch.device | str | None = None,
        architecture: str = "basic",
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be in the range [0, 1)")
        if architecture not in {"basic", "dcgan"}:
            raise ValueError("architecture must be 'basic' or 'dcgan'")
        self.architecture = architecture
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if generator is None:
            generator = DCGANGenerator(latent_dim) if architecture == "dcgan" else BasicGenerator(latent_dim)
        if discriminator is None:
            discriminator = DCGANDiscriminator() if architecture == "dcgan" else BasicDiscriminator()
        self.latent_dim = latent_dim
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=learning_rate, betas=(beta1, 0.999)
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999)
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.history: dict[str, list[float]] = {
            "generator_loss": [],
            "discriminator_loss": [],
        }

    def train_epoch(self, loader: Iterable[tuple[Tensor, Tensor]]) -> dict[str, float]:
        """Run one epoch, updating D on detached fakes and then G."""
        self.generator.train()
        self.discriminator.train()
        generator_losses: list[float] = []
        discriminator_losses: list[float] = []

        for real_images, _ in loader:
            real_images = real_images.to(self.device)
            batch_size = real_images.shape[0]
            real_targets = torch.ones(batch_size, 1, device=self.device)
            fake_targets = torch.zeros(batch_size, 1, device=self.device)

            self.discriminator_optimizer.zero_grad(set_to_none=True)
            fake_images = self.generator(sample_noise(batch_size, self.latent_dim, device=self.device))
            real_loss = self.criterion(self.discriminator(real_images), real_targets)
            fake_loss = self.criterion(
                self.discriminator(fake_images.detach()), fake_targets
            )
            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            self.generator_optimizer.zero_grad(set_to_none=True)
            generated_images = self.generator(
                sample_noise(batch_size, self.latent_dim, device=self.device)
            )
            generator_loss = self.criterion(
                self.discriminator(generated_images), real_targets
            )
            generator_loss.backward()
            self.generator_optimizer.step()

            discriminator_losses.append(float(discriminator_loss.item()))
            generator_losses.append(float(generator_loss.item()))

        if not generator_losses:
            raise ValueError("training loader produced no batches")
        metrics = {
            "generator_loss": sum(generator_losses) / len(generator_losses),
            "discriminator_loss": sum(discriminator_losses) / len(discriminator_losses),
        }
        for name, value in metrics.items():
            self.history[name].append(value)
        return metrics

    def train(
        self,
        loader: Iterable[tuple[Tensor, Tensor]],
        *,
        epochs: int,
        checkpoint_dir: str | Path = "checkpoints",
        output_dir: str | Path = "outputs/training_graphs",
        checkpoint_every: int = 1,
        sample_count: int = 16,
    ) -> dict[str, list[float]]:
        """Train for the requested epochs and save checkpoints/samples."""
        if epochs < 1:
            raise ValueError("epochs must be greater than zero")
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be greater than zero")
        fixed_noise = sample_noise(sample_count, self.latent_dim, device=self.device)

        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(loader)
            LOGGER.info(
                "Epoch %d/%d - generator_loss=%.4f discriminator_loss=%.4f",
                epoch,
                epochs,
                metrics["generator_loss"],
                metrics["discriminator_loss"],
            )
            if epoch % checkpoint_every == 0 or epoch == epochs:
                checkpoint_path = Path(checkpoint_dir) / f"{self.architecture}_gan_checkpoint.pth"
                save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    generator=self.generator,
                    discriminator=self.discriminator,
                    generator_optimizer=self.generator_optimizer,
                    discriminator_optimizer=self.discriminator_optimizer,
                    history=self.history,
                )
                save_history(Path(output_dir) / f"{self.architecture}_gan_history.json", self.history)
                with torch.no_grad():
                    samples = self.generator(fixed_noise)
                save_sample_grid(
                    samples,
                    Path(output_dir) / f"{self.architecture}_gan_epoch_{epoch:03d}.png",
                )
        return self.history
