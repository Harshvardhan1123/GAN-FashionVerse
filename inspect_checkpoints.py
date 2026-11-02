import torch
import os

checkpoint_dir = "checkpoints"

files = [
    "fashion_color_generator_v2.pth",
    "fashion_color_generator_v2_ep60.pth",
    "dcgan_gan_checkpoint_v2.pth"
]

for filename in files:

    path = os.path.join(checkpoint_dir, filename)

    print("\n" + "=" * 60)
    print("CHECKPOINT:", filename)
    print("=" * 60)

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        print("Type:", type(checkpoint))

        if isinstance(checkpoint, dict):

            print("\nMain Keys:")

            for key, value in checkpoint.items():

                if isinstance(value, dict):
                    print(f"\n{key}:")
                    print("  Nested keys:", list(value.keys())[:15])

                elif isinstance(value, torch.Tensor):
                    print(
                        f"{key}: Tensor {tuple(value.shape)}"
                    )

                else:
                    print(f"{key}: {type(value)}")

    except Exception as e:
        print("ERROR:", e)