from __future__ import annotations

import pytest
import torch

from models.discriminator import BasicDiscriminator
from models.generator import BasicGenerator, sample_noise


def test_basic_generator_output_dimensions_and_range() -> None:
    generator = BasicGenerator()
    images = generator(sample_noise(batch_size=4))

    assert images.shape == (4, 1, 28, 28)
    assert torch.all(images >= -1.0)
    assert torch.all(images <= 1.0)


def test_basic_discriminator_returns_one_logit_per_image() -> None:
    discriminator = BasicDiscriminator()

    logits = discriminator(torch.randn(4, 1, 28, 28))

    assert logits.shape == (4, 1)


def test_generator_and_discriminator_are_differentiable() -> None:
    generator = BasicGenerator()
    discriminator = BasicDiscriminator()

    fake_images = generator(sample_noise(batch_size=2))
    loss = discriminator(fake_images).mean()
    loss.backward()

    assert any(parameter.grad is not None for parameter in generator.parameters())
    assert any(
        parameter.grad is not None for parameter in discriminator.parameters()
    )


@pytest.mark.parametrize(
    ("batch_size", "latent_dim"),
    [(0, 100), (2, 0)],
)
def test_sample_noise_rejects_invalid_dimensions(
    batch_size: int, latent_dim: int
) -> None:
    with pytest.raises(ValueError):
        sample_noise(batch_size, latent_dim)


def test_models_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError):
        BasicGenerator()(torch.randn(2, 99))
    with pytest.raises(ValueError):
        BasicDiscriminator()(torch.randn(2, 28, 28))
