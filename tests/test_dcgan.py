from __future__ import annotations

import pytest
import torch

from models.dcgan_discriminator import DCGANDiscriminator
from models.dcgan_generator import DCGANGenerator
from models.generator import sample_noise


def test_dcgan_generator_output_dimensions_and_range() -> None:
    generator = DCGANGenerator()

    images = generator(sample_noise(4))

    assert images.shape == (4, 1, 28, 28)
    assert torch.all(images >= -1.0)
    assert torch.all(images <= 1.0)


def test_dcgan_discriminator_returns_one_logit_per_image() -> None:
    discriminator = DCGANDiscriminator()

    logits = discriminator(torch.randn(4, 1, 28, 28))

    assert logits.shape == (4, 1)


def test_dcgan_models_support_gradient_flow() -> None:
    generator = DCGANGenerator()
    discriminator = DCGANDiscriminator()

    loss = discriminator(generator(sample_noise(2))).mean()
    loss.backward()

    assert any(parameter.grad is not None for parameter in generator.parameters())
    assert any(parameter.grad is not None for parameter in discriminator.parameters())


@pytest.mark.parametrize(
    "model",
    [DCGANGenerator(), DCGANDiscriminator()],
)
def test_dcgan_models_reject_invalid_inputs(model) -> None:
    with pytest.raises(ValueError):
        model(torch.randn(2, 28, 28))
