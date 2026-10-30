from __future__ import annotations

import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from data.dataset import (
    FASHION_MNIST_CLASSES,
    create_fashion_mnist_loader,
    dataset_metadata,
    denormalize_images,
    plot_dataset_samples,
)


def test_dataset_metadata_reports_fashion_mnist_shape() -> None:
    dataset = FashionMNIST.__new__(FashionMNIST)
    dataset.data = torch.zeros((3, 28, 28), dtype=torch.uint8)
    dataset.targets = torch.tensor([0, 1, 2])
    dataset.target_transform = None

    metadata = dataset_metadata(dataset)

    assert metadata["samples"] == 3
    assert metadata["image_dimensions"] == (28, 28)
    assert metadata["channels"] == 1
    assert metadata["classes"] == 10
    assert metadata["class_names"] == FASHION_MNIST_CLASSES


def test_denormalize_images_maps_gan_range_to_display_range() -> None:
    images = torch.tensor([-1.0, 0.0, 1.0, 2.0]).reshape(1, 1, 2, 2)

    result = denormalize_images(images)

    assert torch.equal(result.flatten(), torch.tensor([0.0, 0.5, 1.0, 1.0]))


def test_sample_plot_has_one_axis_per_requested_sample() -> None:
    dataset = FashionMNIST.__new__(FashionMNIST)
    dataset.data = torch.zeros((3, 28, 28), dtype=torch.uint8)
    dataset.targets = torch.tensor([0, 1, 2])
    dataset.target_transform = None
    dataset.transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    figure = plot_dataset_samples(dataset, count=3, columns=2)

    assert len(figure.axes) == 4
    figure.clf()


def test_loader_rejects_invalid_batch_size() -> None:
    try:
        create_fashion_mnist_loader(batch_size=0, download=False)
    except ValueError as error:
        assert str(error) == "batch_size must be greater than zero"
    else:
        raise AssertionError("Expected invalid batch size to raise ValueError")
