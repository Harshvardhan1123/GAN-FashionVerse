"""Adversarial training loop for the independent face GAN."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn

from models.face_generator import FaceDiscriminator, FaceGenerator
from training.utils import save_checkpoint, save_history

LOGGER = logging.getLogger(__name__)


class FaceGANTrainer:
    """Train the face GAN on real RGB face images."""

    def __init__(
        self,
        *,
        latent_dim: int = 100,
        learning_rate: float = 0.0002,
        beta1: float = 0.5,
        device: torch.device | str | None = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.generator = FaceGenerator(latent_dim).to(self.device)
        self.discriminator = FaceDiscriminator().to(self.device)
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

    def train(
        self,
        loader: Iterable[tuple[Tensor, Tensor]],
        *,
        epochs: int = 1,
        checkpoint_path: str | Path = "checkpoints/face_generator.pth",
        history_path: str | Path = "outputs/faces/face_gan_history.json",
    ) -> dict[str, list[float]]:
        """Train and save a face checkpoint after each completed epoch."""
        if epochs < 1:
            raise ValueError("epochs must be greater than zero")
        for epoch in range(1, epochs + 1):
            generator_losses: list[float] = []
            discriminator_losses: list[float] = []
            for real_images, _ in loader:
                real_images = real_images.to(self.device)
                batch_size = real_images.shape[0]
                real_targets = torch.ones(batch_size, 1, device=self.device)
                fake_targets = torch.zeros(batch_size, 1, device=self.device)

                self.discriminator_optimizer.zero_grad(set_to_none=True)
                fake_images = self.generator(
                    torch.randn(batch_size, self.latent_dim, device=self.device)
                )
                discriminator_loss = self.criterion(
                    self.discriminator(real_images), real_targets
                ) + self.criterion(
                    self.discriminator(fake_images.detach()), fake_targets
                )
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                self.generator_optimizer.zero_grad(set_to_none=True)
                generated_images = self.generator(
                    torch.randn(batch_size, self.latent_dim, device=self.device)
                )
                generator_loss = self.criterion(
                    self.discriminator(generated_images), real_targets
                )
                generator_loss.backward()
                self.generator_optimizer.step()
                generator_losses.append(float(generator_loss.item()))
                discriminator_losses.append(float(discriminator_loss.item()))

            if not generator_losses:
                raise ValueError("training loader produced no batches")
            self.history["generator_loss"].append(
                sum(generator_losses) / len(generator_losses)
            )
            self.history["discriminator_loss"].append(
                sum(discriminator_losses) / len(discriminator_losses)
            )
            LOGGER.info("Face epoch %d/%d complete", epoch, epochs)
            save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                generator=self.generator,
                discriminator=self.discriminator,
                generator_optimizer=self.generator_optimizer,
                discriminator_optimizer=self.discriminator_optimizer,
                history=self.history,
            )
            save_history(history_path, self.history)
        return self.history
