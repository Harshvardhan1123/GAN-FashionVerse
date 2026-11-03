"""Explicit face-training entry point using a user-provided image folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.faces import create_face_loader
from training.face_trainer import FaceGANTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the face generator.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="number of training epochs (default: 50)",
    )
    return parser


def main() -> None:
    """Train the independent face GAN on a real image directory."""
    args = build_parser().parse_args()
    dataset, loader = create_face_loader(args.data_dir)
    print(f"Training on {len(dataset)} real face images from {args.data_dir}")
    FaceGANTrainer().train(loader, epochs=args.epochs)


if __name__ == "__main__":
    main()
