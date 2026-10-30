"""Checkpoint-aware face inference with no untrained fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor

from models.face_generator import FaceGenerator


class FaceModelUnavailableError(RuntimeError):
    """Raised when a trained face generator cannot be used."""


def generate_faces(
    checkpoint_path: str | Path = "checkpoints/face_generator.pth",
    *,
    count: int = 1,
    seed: int | None = None,
    latent_dim: int = 100,
    device: torch.device | str | None = None,
) -> Tensor:
    """Generate faces only from a valid trained face checkpoint."""
    if count < 1:
        raise ValueError("count must be greater than zero")
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FaceModelUnavailableError(
            "The face-generation model has not been trained yet; no faces are available."
        )

    active_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    generator = FaceGenerator(latent_dim).to(active_device)
    try:
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_file, map_location=active_device, weights_only=False
        )
        generator.load_state_dict(checkpoint["generator"])
    except (KeyError, RuntimeError, OSError, ValueError) as error:
        raise FaceModelUnavailableError(
            f"Could not load the trained face model: {error}"
        ) from error

    generator.eval()
    rng = torch.Generator(device=active_device)
    if seed is not None:
        rng.manual_seed(seed)
    noise = torch.randn(count, latent_dim, device=active_device, generator=rng)
    with torch.no_grad():
        return generator(noise).cpu()


def save_face_images(
    images: Tensor,
    output_dir: str | Path = "outputs/faces",
    *,
    seed: int | None = None,
) -> list[Path]:
    """Save generated RGB face tensors as PNG files."""
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 64, 64):
        raise ValueError("images must have shape (batch, 3, 64, 64)")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    timestamp = torch.randint(0, 2**31 - 1, (1,)).item()
    for index, image in enumerate(images):
        display = ((image.detach().cpu() + 1.0) / 2.0).clamp(0, 1)
        file_path = output_path / f"face_{timestamp}_{index:03d}.png"
        Image.fromarray(
            display.permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy(),
            mode="RGB",
        ).save(file_path, format="PNG")
        paths.append(file_path)
    return paths
