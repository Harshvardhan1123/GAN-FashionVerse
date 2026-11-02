"""Improved GAN training for GAN-FashionVerse.

Put this file in training/train_improved.py and run from the project root:

    python -m training.train_improved fashion --epochs 50
    python -m training.train_improved faces --data-dir path/to/faces --epochs 60

What is improved compared to the old trainers
  * Hinge loss instead of BCE
  * Spectral normalisation in the discriminator (no BatchNorm there)
  * Two time-scale learning rates (D 4e-4, G 1e-4), betas (0.0, 0.9)
  * EMA copy of the generator; the EMA weights are saved as
    checkpoint["generator"], so the existing app loads them unchanged
  * Faces: centre-square crop (no stretching), horizontal flip
  * Fixed-noise sample grids every epoch, generated in eval mode
  * Resume support and periodic snapshots

The generators (DCGANGenerator, FaceGenerator) are NOT changed, so the
checkpoints stay compatible with inference/generate_*.py.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.dcgan_generator import DCGANGenerator
from models.face_generator import FaceGenerator

LATENT_DIM = 100
G_LR, D_LR = 1e-4, 4e-4
BETAS = (0.0, 0.9)
EMA_DECAY = 0.999


# ----------------------------------------------------------------- data
def center_square(img: Image.Image) -> Image.Image:
    return transforms.functional.center_crop(img, min(img.size))


def pad_square(img: Image.Image) -> Image.Image:
    """Pad to a square on a white background (keeps whole garment visible)."""
    side = max(img.size)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas


class FaceFolder(Dataset):
    """Recursive RGB image folder (used for faces and colour fashion)."""

    EXT = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self, root: str, size: int = 64, crop: int = 0, pad: bool = False
    ) -> None:
        self.paths = sorted(
            p for p in Path(root).rglob("*") if p.suffix.lower() in self.EXT
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")
        if pad:
            first = transforms.Lambda(pad_square)
        elif crop:
            first = transforms.CenterCrop(crop)
        else:
            first = transforms.Lambda(center_square)
        self.tf = transforms.Compose(
            [
                first,
                transforms.Resize(size, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        )
        print(f"Found {len(self.paths)} face images")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), 0


# --------------------------------------------------------------- models
def sn_conv(i, o, k, s, p):
    return spectral_norm(nn.Conv2d(i, o, k, s, p))


def make_discriminator(nc: int, size: int, fm: int = 64) -> nn.Module:
    act = lambda: nn.LeakyReLU(0.1, inplace=True)  # noqa: E731
    if size == 28:
        return nn.Sequential(
            sn_conv(nc, fm, 3, 1, 1), act(),
            sn_conv(fm, fm, 4, 2, 1), act(),          # 14
            sn_conv(fm, fm * 2, 3, 1, 1), act(),
            sn_conv(fm * 2, fm * 2, 4, 2, 1), act(),  # 7
            sn_conv(fm * 2, fm * 4, 3, 1, 1), act(),
            nn.Flatten(),
            spectral_norm(nn.Linear(fm * 4 * 7 * 7, 1)),
        )
    return nn.Sequential(                              # 64 x 64
        sn_conv(nc, fm, 4, 2, 1), act(),               # 32
        sn_conv(fm, fm * 2, 4, 2, 1), act(),           # 16
        sn_conv(fm * 2, fm * 4, 4, 2, 1), act(),       # 8
        sn_conv(fm * 4, fm * 8, 4, 2, 1), act(),       # 4
        nn.Flatten(),
        spectral_norm(nn.Linear(fm * 8 * 4 * 4, 1)),
    )


def init_generator(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for be, b in zip(ema.buffers(), model.buffers()):
        be.copy_(b)


# ------------------------------------------------------------- training
def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.target == "fashion":
        size, nc = 28, 1
        batch = args.batch_size or 128
        epochs = args.epochs or 50
        G = DCGANGenerator(latent_dim=LATENT_DIM)
        tf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        data = datasets.FashionMNIST(
            args.fashion_root, train=True, download=True, transform=tf
        )
        ckpt_path = Path("checkpoints/dcgan_gan_checkpoint_v2.pth")
        out_dir = Path("outputs/improved/fashion")
    else:
        if not args.data_dir:
            raise SystemExit("--data-dir is required for this target")
        size, nc = 64, 3
        batch = args.batch_size or 64
        G = FaceGenerator(latent_dim=LATENT_DIM)  # generic 100 -> 64x64 RGB
        if args.target == "fashion-color":
            epochs = args.epochs or 80
            data = FaceFolder(args.data_dir, size=64, pad=True)
            ckpt_path = Path("checkpoints/fashion_color_generator_v2.pth")
            out_dir = Path("outputs/improved/fashion_color")
        else:
            epochs = args.epochs or 60
            data = FaceFolder(args.data_dir, size=64, crop=args.crop)
            ckpt_path = Path("checkpoints/face_generator_v2.pth")
            out_dir = Path("outputs/improved/faces")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        data,
        batch_size=batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    G.apply(init_generator)
    G, D = G.to(device), make_discriminator(nc, size).to(device)
    ema = copy.deepcopy(G).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt_g = torch.optim.Adam(G.parameters(), lr=G_LR, betas=BETAS)
    opt_d = torch.optim.Adam(D.parameters(), lr=D_LR, betas=BETAS)
    history = {"generator_loss": [], "discriminator_loss": []}
    start = 0

    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(ck["generator_raw"])
        ema.load_state_dict(ck["generator"])
        D.load_state_dict(ck["discriminator"])
        opt_g.load_state_dict(ck["generator_optimizer"])
        opt_d.load_state_dict(ck["discriminator_optimizer"])
        history, start = ck["history"], ck["epoch"]
        print(f"Resumed from epoch {start}")

    fixed_noise = torch.randn(64, LATENT_DIM, device=device)
    step = start * len(loader)

    for epoch in range(start, epochs):
        G.train()
        D.train()
        g_sum = d_sum = 0.0
        for real, _ in loader:
            real = real.to(device, non_blocking=True)
            b = real.shape[0]

            # --- discriminator (hinge loss)
            with torch.no_grad():
                fake = G(torch.randn(b, LATENT_DIM, device=device))
            d_loss = F.relu(1.0 - D(real)).mean() + F.relu(1.0 + D(fake)).mean()
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            # --- generator
            g_loss = -D(G(torch.randn(b, LATENT_DIM, device=device))).mean()
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            step += 1
            update_ema(ema, G, min(EMA_DECAY, (1 + step) / (10 + step)))
            g_sum += g_loss.item()
            d_sum += d_loss.item()

        n = len(loader)
        history["generator_loss"].append(g_sum / n)
        history["discriminator_loss"].append(d_sum / n)
        print(
            f"epoch {epoch + 1}/{epochs}  "
            f"G {g_sum / n:.4f}  D {d_sum / n:.4f}"
        )

        with torch.no_grad():  # samples from EMA generator, eval mode
            ema.eval()
            save_image(
                ema(fixed_noise).cpu(),
                out_dir / f"epoch_{epoch + 1:03d}.png",
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
            )

        state = {
            "epoch": epoch + 1,
            "generator": ema.state_dict(),  # what the app loads
            "generator_raw": G.state_dict(),
            "discriminator": D.state_dict(),
            "generator_optimizer": opt_g.state_dict(),
            "discriminator_optimizer": opt_d.state_dict(),
            "history": history,
        }
        torch.save(state, ckpt_path)
        if (epoch + 1) % 10 == 0:
            torch.save(state, ckpt_path.with_name(
                f"{ckpt_path.stem}_ep{epoch + 1}.pth"))
        (out_dir / "history.json").write_text(json.dumps(history))

    print("Done. Checkpoint:", ckpt_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("target", choices=["fashion", "faces", "fashion-color"])
    p.add_argument("--data-dir", default=None, help="face image folder")
    p.add_argument("--fashion-root", default="datasets/fashion_mnist")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--crop", type=int, default=0,
                   help="optional centre crop before resize (CelebA: 140)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()