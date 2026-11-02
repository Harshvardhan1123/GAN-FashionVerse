import torch

files = [
    "fashion_color_generator_v2.pth",
    "fashion_color_generator_v2_ep60.pth",
    "dcgan_gan_checkpoint_v2.pth"
]

for filename in files:

    print("\n" + "=" * 60)
    print(filename)
    print("=" * 60)

    checkpoint = torch.load(
        f"checkpoints/{filename}",
        map_location="cpu",
        weights_only=True
    )

    print("Epoch:", checkpoint.get("epoch"))

    for model_name in ["generator", "generator_raw", "discriminator"]:

        if model_name not in checkpoint:
            continue

        print(f"\n{model_name.upper()} PARAMETERS:")

        state = checkpoint[model_name]

        for key, value in state.items():

            if isinstance(value, torch.Tensor):
                print(f"{key}: {tuple(value.shape)}")

    print("\nTRAINING HISTORY:")

    history = checkpoint.get("history", {})

    for key, value in history.items():
        print(f"{key}: {len(value)} entries")