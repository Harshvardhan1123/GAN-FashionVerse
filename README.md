# GAN FashionVerse

**Where Artificial Intelligence Meets Fashion Creativity.**

GAN FashionVerse is an academic deep-learning project that demonstrates how
Generative Adversarial Networks can learn to synthesize Fashion-MNIST images.
The project includes both a fully connected Basic GAN and an independent
convolutional DCGAN, with a modular PyTorch training pipeline and Streamlit
interface.

## Project status

Implemented:
 
- Fashion-MNIST loading, normalization, metadata, and visualization
- Basic MLP GAN with a 100-dimensional latent vector
- DCGAN for 28x28 grayscale images
- BCE-with-logits adversarial training
- CPU fallback and CUDA detection
- Checkpoint and loss-history persistence
- Seeded inference and PNG gallery export
- Streamlit dashboard and training visualizations
- Separate face-GAN architecture with honest untrained-model handling

The face module is trained separately from the fashion models. It uses a real
face dataset and does not display faces until a valid face checkpoint exists.

## Technology stack

- Python 3.11
- PyTorch and Torchvision
- NumPy and Pillow
- Matplotlib
- Streamlit
- Pytest

## GAN architectures

### Basic GAN

The Basic GAN uses fully connected networks:

- Generator: `100 -> 256 -> 512 -> 1024 -> 784`, ReLU and Tanh
- Discriminator: `784 -> 512 -> 256 -> 1`, LeakyReLU and logits

The generator output is reshaped to `1 x 28 x 28`.

### DCGAN

The DCGAN uses transposed convolutions in the generator and convolutional
layers in the discriminator, with batch normalization, ReLU/LeakyReLU
activations, Tanh output, and DCGAN-style weight initialization.
Both models use `BCEWithLogitsLoss`. The discriminator receives detached fake
images during its update, while the generator receives a separate update to
fool the discriminator.

## Dataset

The initial dataset is Fashion-MNIST:

- 60,000 training images and 10,000 test images
- 10 clothing classes
- 28 x 28 grayscale images
- Values normalized to `[-1, 1]`
- Stored under `datasets/fashion_mnist/`

The model generates small grayscale clothing concepts. It does not generate
photorealistic full-body outfits.

## Installation

From the project root on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Check the environment:

```powershell
python check_gpu.py
```

CUDA is used automatically when available. Otherwise, training falls back to
CPU.

## Training

Train the Basic GAN and download Fashion-MNIST on first use:

```powershell
python -m training.train --architecture basic --epochs 10 --download
```

Train the DCGAN:

```powershell
python -m training.train --architecture dcgan --epochs 10 --download
```

Training writes checkpoints to `checkpoints/` and histories/samples to
`outputs/training_graphs/`. The default batch size is 128 and can be changed
in the training entry point or configuration.

## Generate images

After training, use the Streamlit Fashion Studio or call inference directly:

```powershell
python -c "from inference.generate_fashion import generate_fashion_images, save_fashion_images; images = generate_fashion_images('checkpoints/basic_gan_checkpoint.pth', count=4, seed=42); print(save_fashion_images(images, seed=42))"
```

Generated PNGs and gallery metadata are stored under `outputs/fashion/`.
Using the same seed and checkpoint produces reproducible results.

## Face model training

The face module supports a recursive image folder and was trained here on the
Kaggle Flickr-Faces-HQ (FFHQ) dataset. After downloading it with KaggleHub,
train with:

```powershell
python -m training.train_faces --data-dir "<path-to>\flickrfaceshq-dataset-ffhq\versions\1" --epochs 1
```

The trained face checkpoint is saved to `checkpoints/face_generator.pth`, with
loss history in `outputs/faces/face_gan_history.json`. The AI Model Studio only
displays faces after this checkpoint exists.

## Start the application
 
```powershell
streamlit run app/app.py
```

The dashboard includes:

1. Home
2. Fashion Studio
3. AI Model Studio
4. Training Dashboard
5. Dataset Explorer
6. How GAN Works
7. About Project

## Project structure

```text
GAN-FashionVerse/
├── app/                  # Streamlit pages and shared UI
├── data/                 # Fashion-MNIST loading and transforms
├── models/               # Basic GAN, DCGAN, and face architectures
├── training/             # Training loops and checkpoint utilities
├── inference/            # Fashion and face inference
├── datasets/             # Downloaded datasets (ignored by Git)
├── checkpoints/          # Model checkpoints (ignored by Git)
├── outputs/              # Generated samples and histories
├── tests/                # Automated tests
├── config.py
├── check_gpu.py
└── requirements.txt
```

## Testing

Run the complete test suite:

```powershell
python -m pytest tests -q
```

The tests cover data utilities, model dimensions, gradients, checkpoint
round-trips, inference reproducibility, DCGAN integration, and face-model
availability behavior.

## Screenshots

Screenshots can be added here after running the Streamlit application locally.
Generated images and training charts should be captured from real checkpoints
and real training history rather than placeholders.

## Future enhancements

- Complete DCGAN-specific training controls in the UI
- Add a real, appropriately licensed face dataset pipeline
- Implement face-GAN training and evaluation
- Add experiment metadata and architecture comparison charts
- Add FID or related evaluation when the dataset and compute budget support it

## Contributors

GAN FashionVerse is an academic project by Harshvardhan1123.









