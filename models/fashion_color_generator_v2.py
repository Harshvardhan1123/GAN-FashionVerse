import torch
from torch import nn


class FashionColorGeneratorV2(nn.Module):

    def __init__(self, latent_dim=100):
        super().__init__()

        self.latent_dim = latent_dim

        self.network = nn.Sequential(

            # 100 -> 512 x 4 x 4
            nn.ConvTranspose2d(
                latent_dim, 512, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 512 -> 256 x 8 x 8
            nn.ConvTranspose2d(
                512, 256, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 256 -> 128 x 16 x 16
            nn.ConvTranspose2d(
                256, 128, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 128 -> 64 x 32 x 32
            nn.ConvTranspose2d(
                128, 64, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64 -> RGB 64 x 64
            nn.ConvTranspose2d(
                64, 3, 4, 2, 1, bias=False
            ),

            nn.Tanh()
        )

    def forward(self, noise):

        if noise.ndim == 2:
            noise = noise.view(
                noise.shape[0],
                self.latent_dim,
                1,
                1
            )

        return self.network(noise)