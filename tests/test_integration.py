from __future__ import annotations

import torch

from config import FashionConfig
from inference.generate_fashion import generate_fashion_images
from models.dcgan_discriminator import DCGANDiscriminator
from models.dcgan_generator import DCGANGenerator
from training.trainer import BasicGANTrainer
from training.utils import save_checkpoint


def test_config_selects_architecture_paths() -> None:
    config = FashionConfig(architecture="dcgan")

    assert config.checkpoint_path.name == "dcgan_gan_checkpoint.pth"
    assert config.history_path.name == "dcgan_gan_history.json"


def test_dcgan_trainer_completes_one_synthetic_batch(tmp_path) -> None:
    trainer = BasicGANTrainer(
        generator=DCGANGenerator(),
        discriminator=DCGANDiscriminator(),
        architecture="dcgan",
        device="cpu",
    )
    images = torch.randn(2, 1, 28, 28)

    history = trainer.train(
        [(images, torch.zeros(2, dtype=torch.long))],
        epochs=1,
        checkpoint_dir=tmp_path / "checkpoints",
        output_dir=tmp_path / "outputs",
        sample_count=2,
    )

    assert len(history["generator_loss"]) == 1
    assert (tmp_path / "checkpoints" / "dcgan_gan_checkpoint.pth").is_file()


def test_dcgan_checkpoint_is_supported_by_inference(tmp_path) -> None:
    generator = DCGANGenerator()
    discriminator = DCGANDiscriminator()
    path = tmp_path / "dcgan.pth"
    save_checkpoint(
        path,
        epoch=1,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=torch.optim.Adam(generator.parameters()),
        discriminator_optimizer=torch.optim.Adam(discriminator.parameters()),
        history={"generator_loss": [1.0], "discriminator_loss": [1.0]},
    )

    images = generate_fashion_images(path, count=1, seed=3, architecture="dcgan", device="cpu")

    assert images.shape == (1, 1, 28, 28)
