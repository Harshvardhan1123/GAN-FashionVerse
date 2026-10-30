"""Command-line entry point for training either fashion GAN architecture."""

from __future__ import annotations

import argparse
import logging

from config import FashionConfig
from data.dataset import create_fashion_mnist_loader
from training.trainer import BasicGANTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Fashion-MNIST GAN.")
    parser.add_argument("--architecture", choices=("basic", "dcgan"), default="basic")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--download", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    config = FashionConfig(
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    loader = create_fashion_mnist_loader(
        config.dataset_root,
        batch_size=config.batch_size,
        download=args.download,
        pin_memory=False,
    )
    trainer = BasicGANTrainer(
        latent_dim=config.latent_dim,
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        architecture=config.architecture,
    )
    trainer.train(
        loader,
        epochs=config.epochs,
        checkpoint_dir=config.checkpoint_dir,
        output_dir=config.output_dir,
    )


if __name__ == "__main__":
    main()
