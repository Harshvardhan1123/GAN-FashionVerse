import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from models.dcgan_generator import DCGANGenerator


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Initialize Generator
generator = DCGANGenerator(
    latent_dim=100,
    feature_maps=64
)


# Load checkpoint
checkpoint = torch.load(
    "checkpoints/dcgan_gan_checkpoint_v2.pth",
    map_location=device,
    weights_only=True
)

print("Training epoch:", checkpoint["epoch"])


# Load trained weights
generator.load_state_dict(
    checkpoint["generator"],
    strict=True
)

generator = generator.to(device)
generator.eval()

print("\nDCGAN V2 Generator loaded successfully!")


# Generate images
noise = torch.randn(4, 100, device=device)

with torch.no_grad():
    generated_images = generator(noise)


print("Generated image shape:", generated_images.shape)


# Normalize images
generated_images = (generated_images + 1) / 2

# Save images
vutils.save_image(
    generated_images,
    "dcgan_v2_generated.png",
    nrow=2
)

print("Images saved as dcgan_v2_generated.png")

print("\nDCGAN V2 is working!")