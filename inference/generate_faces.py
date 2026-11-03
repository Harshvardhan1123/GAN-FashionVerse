"""Checkpoint-aware face inference with no untrained fallback."""

from __future__ import annotations

import sys
from functools import lru_cache
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from models.face_generator import FaceGenerator


class FaceModelUnavailableError(RuntimeError):
    """Raised when a trained face generator cannot be used."""


STYLEGAN_RUNTIME = Path(__file__).resolve().parent.parent / "third_party" / "stylegan"


@lru_cache(maxsize=2)
def _load_stylegan_generator(
    checkpoint_path: str, device_name: str
) -> tuple[Any, Any, Any]:
    """Load and cache the original TensorFlow StyleGAN generator."""
    if str(STYLEGAN_RUNTIME) not in sys.path:
        sys.path.insert(0, str(STYLEGAN_RUNTIME))
    try:
        import tensorflow as tf

        tf.compat.v1.disable_eager_execution()
        for name in dir(tf.compat.v1):
            if not hasattr(tf, name):
                setattr(tf, name, getattr(tf.compat.v1, name))
        tf.name_scope = tf.compat.v1.name_scope
        shape_getitem = tf.TensorShape.__getitem__

        def legacy_shape_getitem(shape: Any, index: Any) -> Any:
            value = shape_getitem(shape, index)
            if isinstance(value, int):
                return tf.compat.v1.Dimension(value)
            return value

        shape_iter = tf.TensorShape.__iter__

        def legacy_shape_iter(shape: Any) -> Any:
            for value in shape_iter(shape):
                yield (
                    tf.compat.v1.Dimension(value)
                    if value is None or isinstance(value, int)
                    else value
                )

        tf.TensorShape.__getitem__ = legacy_shape_getitem
        tf.TensorShape.__iter__ = legacy_shape_iter
        import dnnlib.tflib as tflib

        tflib.init_tf()
        with Path(checkpoint_path).open("rb") as checkpoint_file:
            _, _, generator = pickle.load(checkpoint_file)
        session = tf.get_default_session()
        if session is None:
            raise RuntimeError("TensorFlow did not create a default session")
    except (ImportError, KeyError, OSError, RuntimeError, ValueError, TypeError) as error:
        raise FaceModelUnavailableError(
            f"Could not load the StyleGAN face model: {error}"
        ) from error
    return generator, session, tf


def _generate_stylegan_faces(
    checkpoint_path: Path,
    *,
    count: int,
    seed: int | None,
    device: torch.device,
) -> Tensor:
    generator, session, tf = _load_stylegan_generator(
        str(checkpoint_path), str(device)
    )
    rng = np.random.RandomState(seed)
    latents = rng.randn(count, generator.input_shape[1])
    with generator.input_templates[0].graph.as_default(), session.as_default():
        output = generator.run(
            latents,
            None,
            truncation_psi=0.7,
            randomize_noise=False,
            output_transform={
                "func": __import__(
                    "dnnlib.tflib", fromlist=["convert_images_to_uint8"]
                ).convert_images_to_uint8,
                "nchw_to_nhwc": True,
            },
        )
    return (torch.from_numpy(output).permute(0, 3, 1, 2).float() / 127.5) - 1.0


def generate_faces(
    checkpoint_path: str | Path = "checkpoints/karras2019stylegan-ffhq-1024x1024.pkl",
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
    if checkpoint_file.suffix.lower() == ".pkl":
        return _generate_stylegan_faces(
            checkpoint_file,
            count=count,
            seed=seed,
            device=active_device,
        )

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
    """Save generated RGB face tensors of any square resolution as PNG files."""
    if (
        images.ndim != 4
        or images.shape[1] != 3
        or images.shape[2] != images.shape[3]
    ):
        raise ValueError("images must have shape (batch, 3, height, width)")
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
