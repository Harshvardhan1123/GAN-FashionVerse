"""Central project configuration for reproducible fashion experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FashionConfig:
    """Runtime defaults shared by training and inference entry points."""

    architecture: str = "basic"
    latent_dim: int = 100
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 0.0002
    beta1: float = 0.5
    dataset_root: Path = Path("datasets/fashion_mnist")
    checkpoint_dir: Path = Path("checkpoints")
    output_dir: Path = Path("outputs/training_graphs")

    def __post_init__(self) -> None:
        if self.architecture not in {"basic", "dcgan"}:
            raise ValueError("architecture must be 'basic' or 'dcgan'")
        if self.latent_dim < 1 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("latent_dim, batch_size, and epochs must be greater than zero")

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f"{self.architecture}_gan_checkpoint.pth"

    @property
    def history_path(self) -> Path:
        return self.output_dir / f"{self.architecture}_gan_history.json"
