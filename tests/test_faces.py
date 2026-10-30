from __future__ import annotations

import pytest
import torch

from inference.generate_faces import FaceModelUnavailableError, generate_faces
from models.face_generator import FaceDiscriminator, FaceGenerator
from data.faces import FaceImageDataset


def test_face_generator_and_discriminator_shapes() -> None:
    generator = FaceGenerator()
    discriminator = FaceDiscriminator()

    faces = generator(torch.randn(2, 100))
    logits = discriminator(faces)

    assert faces.shape == (2, 3, 64, 64)
    assert logits.shape == (2, 1)


def test_face_models_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError):
        FaceGenerator()(torch.randn(2, 99))
    with pytest.raises(ValueError):
        FaceDiscriminator()(torch.randn(2, 1, 64, 64))


def test_face_inference_does_not_fabricate_untrained_faces(tmp_path) -> None:
    with pytest.raises(FaceModelUnavailableError, match="has not been trained"):
        generate_faces(tmp_path / "missing.pth", device="cpu")


def test_face_dataset_loads_recursive_images(tmp_path) -> None:
    from PIL import Image

    image_path = tmp_path / "real" / "sample.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (80, 60), color="white").save(image_path)

    dataset = FaceImageDataset(tmp_path / "real")

    image, label = dataset[0]
    assert len(dataset) == 1
    assert image.shape == (3, 64, 64)
    assert label == 0
