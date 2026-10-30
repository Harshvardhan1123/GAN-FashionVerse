from __future__ import annotations

import json

import pytest
import torch

from inference.generate_fashion import generate_fashion_images, save_fashion_images
from models.discriminator import BasicDiscriminator
from models.generator import BasicGenerator
from training.utils import save_checkpoint


def _checkpoint(path) -> None:
    generator = BasicGenerator()
    discriminator = BasicDiscriminator()
    save_checkpoint(
        path,
        epoch=1,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=torch.optim.Adam(generator.parameters()),
        discriminator_optimizer=torch.optim.Adam(discriminator.parameters()),
        history={"generator_loss": [1.0], "discriminator_loss": [1.0]},
    )


def test_seeded_generation_is_reproducible(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    _checkpoint(checkpoint)

    first = generate_fashion_images(checkpoint, count=2, seed=42, device="cpu")
    second = generate_fashion_images(checkpoint, count=2, seed=42, device="cpu")

    assert first.shape == (2, 1, 28, 28)
    assert torch.equal(first, second)


def test_save_fashion_images_creates_pngs_and_gallery_metadata(tmp_path) -> None:
    paths = save_fashion_images(
        torch.zeros(2, 1, 28, 28),
        tmp_path,
        seed=7,
    )

    assert len(paths) == 2
    assert all(path.suffix == ".png" and path.is_file() for path in paths)
    gallery = json.loads((tmp_path / "gallery.json").read_text(encoding="utf-8"))
    assert len(gallery) == 2
    assert {record["seed"] for record in gallery} == {7}


def test_inference_reports_missing_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        generate_fashion_images(tmp_path / "missing.pth")
