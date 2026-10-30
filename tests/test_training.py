from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.discriminator import BasicDiscriminator
from models.generator import BasicGenerator
from training.trainer import BasicGANTrainer
from training.utils import load_checkpoint, save_checkpoint, save_history


def test_basic_trainer_records_metrics_and_saves_artifacts(tmp_path) -> None:
    images = torch.randn(4, 1, 28, 28)
    loader = DataLoader(TensorDataset(images, torch.zeros(4, dtype=torch.long)), batch_size=2)
    trainer = BasicGANTrainer(device="cpu")

    history = trainer.train(
        loader,
        epochs=1,
        checkpoint_dir=tmp_path / "checkpoints",
        output_dir=tmp_path / "outputs",
        sample_count=4,
    )

    assert len(history["generator_loss"]) == 1
    assert len(history["discriminator_loss"]) == 1
    assert (tmp_path / "checkpoints" / "basic_gan_checkpoint.pth").is_file()
    assert (tmp_path / "outputs" / "basic_gan_history.json").is_file()
    assert (tmp_path / "outputs" / "basic_gan_epoch_001.png").is_file()


def test_checkpoint_round_trip_restores_epoch_and_history(tmp_path) -> None:
    generator = BasicGenerator()
    discriminator = BasicDiscriminator()
    generator_optimizer = torch.optim.Adam(generator.parameters())
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters())
    history = {"generator_loss": [1.2], "discriminator_loss": [0.8]}
    path = save_checkpoint(
        tmp_path / "model.pth",
        epoch=3,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        history=history,
    )

    loaded = load_checkpoint(
        path,
        generator=BasicGenerator(),
        discriminator=BasicDiscriminator(),
        map_location="cpu",
    )

    assert loaded["epoch"] == 3
    assert loaded["history"] == history


def test_history_is_valid_json(tmp_path) -> None:
    path = save_history(tmp_path / "history.json", {"generator_loss": [1.0]})

    assert json.loads(path.read_text(encoding="utf-8")) == {"generator_loss": [1.0]}


def test_empty_loader_is_rejected() -> None:
    trainer = BasicGANTrainer(device="cpu")
    with pytest.raises(ValueError, match="no batches"):
        trainer.train_epoch([])
