import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from models.fashion_color_generator_v2 import (
    FashionColorGeneratorV2
)


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Initialize model
generator = FashionColorGeneratorV2(latent_dim=100)


# Load checkpoint
checkpoint_path = (
    "checkpoints/fashion_color_generator_v2.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

print("Training epoch:", checkpoint["epoch"])


# Load trained Generator weights
generator.load_state_dict(
    checkpoint["generator"],
    strict=True
)

generator = generator.to(device)
generator.eval()


print("\nGenerator loaded successfully!")


# Generate images
noise = torch.randn(4, 100, device=device)

with torch.no_grad():

    generated_images = generator(noise)


# Convert generated images from [-1, 1] to [0, 1]
generated_images = (generated_images + 1) / 2

# Save generated images
vutils.save_image(
    generated_images,
    "fashion_v2_generated.png",
    nrow=2
)

print("Images saved as fashion_v2_generated.png")


print("Generated image shape:", generated_images.shape)

print("Minimum pixel value:", generated_images.min().item())

print("Maximum pixel value:", generated_images.max().item())

print("\nFashion Color Generator V2 is working!")