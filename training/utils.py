"""Persistence and device helpers for GAN training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

LOGGER = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Prefer CUDA when available and otherwise use the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    history: dict[str, list[float]],
) -> Path:
    """Save model, optimizer, epoch, and metric state for resuming."""
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "history": history,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: torch.optim.Optimizer | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    map_location: torch.device | str | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore optimizer state."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        if generator_optimizer is not None:
            generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        if discriminator_optimizer is not None:
            discriminator_optimizer.load_state_dict(
                checkpoint["discriminator_optimizer"]
            )
    except (KeyError, RuntimeError, OSError, ValueError) as error:
        raise RuntimeError(f"Could not load checkpoint {checkpoint_path}: {error}") from error
    return checkpoint


def save_history(path: str | Path, history: dict[str, list[float]]) -> Path:
    """Persist training metrics as readable JSON."""
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history_path


def save_sample_grid(images: Tensor, path: str | Path, *, columns: int = 4) -> Path:
    """Save generated images as a displayable PNG grid."""
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError("images must have shape (batch, 1, 28, 28)")
    if columns < 1:
        raise ValueError("columns must be greater than zero")

    image_count = len(images)
    rows = (image_count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2))
    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    display_images = ((images.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    for index, image in enumerate(display_images):
        flat_axes[index].imshow(image.squeeze(0), cmap="gray", vmin=0.0, vmax=1.0)
        flat_axes[index].axis("off")
    for axis in flat_axes[image_count:]:
        axis.axis("off")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path
