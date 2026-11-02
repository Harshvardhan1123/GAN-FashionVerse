"""Generate and persist Basic GAN fashion samples."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor

from models.dcgan_generator import DCGANGenerator
from models.fashion_color_generator_v2 import FashionColorGeneratorV2
from models.generator import BasicGenerator, sample_noise


def generate_fashion_images(
    checkpoint_path: str | Path,
    *,
    count: int = 1,
    seed: int | None = None,
    latent_dim: int = 100,
    device: torch.device | str | None = None,
    architecture: str = "basic",
) -> Tensor:
    """Load a trained generator and return reproducible normalized images."""
    if count < 1:
        raise ValueError("count must be greater than zero")
    if seed is not None and seed < 0:
        raise ValueError("seed cannot be negative")
    if architecture not in {"basic", "dcgan", "color_v2"}:
        raise ValueError("architecture must be 'basic', 'dcgan', or 'color_v2'")

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Generator checkpoint does not exist: {checkpoint_file}")

    active_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if architecture == "dcgan":
        generator = DCGANGenerator(latent_dim)
    elif architecture == "color_v2":
        generator = FashionColorGeneratorV2(latent_dim)
    else:
        generator = BasicGenerator(latent_dim)
    generator = generator.to(active_device)
    try:
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_file,
            map_location=active_device,
            weights_only=False,
        )
        generator.load_state_dict(checkpoint["generator"])
    except (KeyError, RuntimeError, OSError, ValueError) as error:
        raise RuntimeError(
            f"Could not load generator checkpoint {checkpoint_file}: {error}"
        ) from error

    if seed is not None:
        generator_seed = torch.Generator(device=active_device)
        generator_seed.manual_seed(seed)
        noise = torch.randn(
            count, latent_dim, device=active_device, generator=generator_seed
        )
    else:
        noise = sample_noise(count, latent_dim, device=active_device)

    generator.eval()
    with torch.no_grad():
        return generator(noise).cpu()


def save_fashion_images(
    images: Tensor,
    output_dir: str | Path = "outputs/fashion",
    *,
    prefix: str = "fashion",
    seed: int | None = None,
) -> list[Path]:
    """Save normalized generated images as individual PNG files."""
    if images.ndim != 4 or tuple(images.shape[1:]) not in {
        (1, 28, 28),
        (3, 64, 64),
    }:
        raise ValueError(
            "images must have shape (batch, 1, 28, 28) or (batch, 3, 64, 64)"
        )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    display_images = ((images.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    paths: list[Path] = []
    for index, image in enumerate(display_images):
        path = output_path / f"{prefix}_{timestamp}_{index:03d}.png"
        if image.shape[0] == 1:
            image_uint8 = (image.squeeze(0).mul(255).round()).to(torch.uint8).numpy()
            Image.fromarray(image_uint8, mode="L").save(path, format="PNG")
        else:
            image_uint8 = image.permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
            Image.fromarray(image_uint8, mode="RGB").save(path, format="PNG")
        paths.append(path)

    _append_gallery_records(
        output_path / "gallery.json",
        paths,
        seed=seed,
    )
    return paths


def _append_gallery_records(
    gallery_path: Path,
    image_paths: list[Path],
    *,
    seed: int | None,
) -> None:
    """Append generated image metadata while preserving an existing gallery."""
    records: list[dict[str, object]] = []
    if gallery_path.exists():
        try:
            existing = json.loads(gallery_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Could not read gallery metadata {gallery_path}: {error}") from error
        if not isinstance(existing, list):
            raise RuntimeError(f"Gallery metadata must contain a list: {gallery_path}")
        records = [record for record in existing if isinstance(record, dict)]

    generated_at = datetime.now(timezone.utc).isoformat()
    records.extend(
        {
            "path": str(path),
            "generated_at": generated_at,
            "seed": seed,
        }
        for path in image_paths
    )
    gallery_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
