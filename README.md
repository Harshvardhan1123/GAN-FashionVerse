# GAN FashionVerse

GAN FashionVerse is a PyTorch project for training and serving GANs that
generate Fashion-MNIST clothing images and optional 64x64 RGB face images.
It includes:

- A fully connected Basic GAN for 28x28 grayscale Fashion-MNIST images.
- A convolutional DCGAN for 28x28 grayscale Fashion-MNIST images.
- An independent 64x64 RGB face GAN.
- An improved trainer with hinge loss, spectral normalization, EMA, resume
  support, and improved face preprocessing.
- A Streamlit application for generation, dataset inspection, and training
  history visualization.

The Fashion-MNIST models generate small grayscale clothing concepts. They do
not generate photorealistic full-body fashion images.

## What is included in a fork

The source code required to train and run the project is stored in these
directories and files:

```text
app/                  Streamlit application and pages
data/                 Dataset loaders and preprocessing
inference/            Checkpoint loading and image generation
models/               Basic, DCGAN, and face model definitions
training/             Training entry points and checkpoint utilities
tests/                Automated tests
config.py             Fashion training defaults
check_gpu.py          Python, PyTorch, and CUDA diagnostics
requirements.txt      Python dependency constraints
```

Generated datasets, checkpoints, and outputs are ignored by Git. A fresh fork
therefore starts without trained checkpoint files and must train locally or
copy compatible checkpoints into `checkpoints/`.

The repository currently contains these training scripts:

```text
training/train.py             Original Fashion-MNIST trainer
training/train_faces.py       Original face trainer
training/train_improved.py    Improved fashion/face trainer
```

## Requirements

The project is configured for Python 3.11. The dependency constraints are:

```text
numpy>=1.26,<3
pillow>=10,<14
matplotlib>=3.8,<4
torch>=2.2,<3
torchvision>=0.17,<1
streamlit>=1.35,<2
plotly>=5.20,<7
pytest>=8,<9
kagglehub>=0.3,<1
```

CUDA is optional. The code automatically uses CUDA when
`torch.cuda.is_available()` is true; otherwise it uses the CPU.

## 1. Clone and create an environment

From PowerShell:

```powershell
git clone https://github.com/Harshvardhan1123/GAN-FashionVerse.git
cd GAN-FashionVerse

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS, use the equivalent `python3 -m venv .venv`,
`source .venv/bin/activate`, and `python -m pip install -r requirements.txt`.

## 2. Verify Python, PyTorch, and CUDA

```powershell
python check_gpu.py
```

The script prints the Python version, platform, PyTorch version, CUDA
availability, and GPU name when CUDA is available.

## 3. Run the tests

```powershell
python -m pytest tests -q
```

The tests cover data utilities, model dimensions, gradients, checkpoint
round-trips, seeded inference, DCGAN integration, and face-model availability.

## 4. Train the original Fashion GANs

Fashion-MNIST is downloaded by Torchvision when `--download` is supplied.
The dataset is stored under `datasets/fashion_mnist/`.

### Basic GAN

```powershell
python -m training.train --architecture basic --epochs 10 --download
```

Optional batch-size override:

```powershell
python -m training.train --architecture basic --epochs 10 --batch-size 128 --download
```

### DCGAN

```powershell
python -m training.train --architecture dcgan --epochs 10 --download
```

The original trainer uses:

```text
Latent dimension: 100
Batch size: 128
Loss: BCEWithLogitsLoss
Optimizer: Adam
Learning rate: 0.0002 for G and D
Betas: (0.5, 0.999)
D updates per batch: 1
G updates per batch: 1
```

Original Fashion checkpoints:

```text
checkpoints/basic_gan_checkpoint.pth
checkpoints/dcgan_gan_checkpoint.pth
```

Original training histories and sample grids:

```text
outputs/training_graphs/basic_gan_history.json
outputs/training_graphs/dcgan_gan_history.json
outputs/training_graphs/basic_gan_epoch_XXX.png
outputs/training_graphs/dcgan_gan_epoch_XXX.png
```

## 5. Train with the improved trainer

`training/train_improved.py` provides an alternate training workflow. It does
not change the generator definitions used by inference. It uses:

```text
Fashion target: Fashion-MNIST DCGAN generator
Face target: FaceGenerator
Loss: hinge loss
Discriminator: spectral normalization, no discriminator BatchNorm
Generator learning rate: 1e-4
Discriminator learning rate: 4e-4
Adam betas: (0.0, 0.9)
EMA decay: 0.999 maximum, with warm-up adjustment
Fashion default epochs: 50
Face default epochs: 60
Fashion default batch size: 128
Face default batch size: 64
DataLoader drop_last: True
```

### Improved Fashion training

```powershell
python -m training.train_improved fashion --epochs 50
```

Useful options:

```powershell
python -m training.train_improved fashion `
  --fashion-root datasets/fashion_mnist `
  --epochs 50 `
  --batch-size 128 `
  --num-workers 2
```

The improved Fashion trainer downloads Fashion-MNIST automatically through
Torchvision and saves:

```text
checkpoints/dcgan_gan_checkpoint_v2.pth
outputs/improved/fashion/history.json
outputs/improved/fashion/epoch_XXX.png
```

Resume an improved Fashion run:

```powershell
python -m training.train_improved fashion --epochs 100 --resume
```

The `--epochs` value is the target ending epoch when resuming.

### Improved face training

The face trainer accepts a folder containing `.jpg`, `.jpeg`, `.png`, or
`.webp` images, including nested folders. The dataset itself is not included
in this repository. Use a dataset whose license permits your intended use.

```powershell
python -m training.train_improved faces `
  --data-dir "C:\path\to\face-images" `
  --epochs 60 `
  --batch-size 64
```

The improved face preprocessing:

- Converts images to RGB.
- Applies a center-square crop by default, avoiding direct aspect-ratio
  stretching.
- Resizes to 64x64 with antialiasing.
- Applies random horizontal flip.
- Normalizes each RGB channel with mean `0.5` and standard deviation `0.5`.

For datasets where a fixed center crop is appropriate:

```powershell
python -m training.train_improved faces `
  --data-dir "C:\path\to\face-images" `
  --crop 140 `
  --epochs 60
```

Improved face outputs:

```text
checkpoints/face_generator_v2.pth
outputs/improved/faces/history.json
outputs/improved/faces/epoch_XXX.png
```

Resume an improved face run:

```powershell
python -m training.train_improved faces `
  --data-dir "C:\path\to\face-images" `
  --epochs 120 `
  --resume
```

The improved checkpoints contain the EMA generator under the existing
`checkpoint["generator"]` key, plus the raw generator under
`checkpoint["generator_raw"]`. The app's generator inference code loads the
EMA generator.

## 6. Run inference from Python

### Fashion inference

For an original Basic GAN checkpoint:

```powershell
python -c "from inference.generate_fashion import generate_fashion_images, save_fashion_images; images = generate_fashion_images('checkpoints/basic_gan_checkpoint.pth', count=4, seed=42, architecture='basic'); print(save_fashion_images(images, seed=42))"
```

For an original or improved DCGAN generator checkpoint:

```powershell
python -c "from inference.generate_fashion import generate_fashion_images, save_fashion_images; images = generate_fashion_images('checkpoints/dcgan_gan_checkpoint.pth', count=4, seed=42, architecture='dcgan'); print(save_fashion_images(images, seed=42))"
```

The same checkpoint and seed produce reproducible fashion latent vectors.
Generated fashion PNGs and `gallery.json` are saved under
`outputs/fashion/`.

### Face inference

```powershell
python -c "from inference.generate_faces import generate_faces, save_face_images; images = generate_faces('checkpoints/face_generator.pth', count=4, seed=42); print(save_face_images(images, seed=42))"
```

For the improved face checkpoint, use:

```powershell
python -c "from inference.generate_faces import generate_faces, save_face_images; images = generate_faces('checkpoints/face_generator_v2.pth', count=4, seed=42); print(save_face_images(images, seed=42))"
```

The face generator returns 64x64 RGB tensors and the saved images are written
under `outputs/faces/`.

## 7. Run the Streamlit application

Start from the repository root:

```powershell
streamlit run app/app.py
```

The project configures Streamlit to bind to `127.0.0.1`. Open the localhost
URL printed by Streamlit.

Available pages:

1. Home
2. Fashion Studio
3. AI Model Studio
4. Training Dashboard
5. Dataset Explorer
6. How GAN Works
7. About Project

### Checkpoint filenames expected by the app

The Fashion Studio expects:

```text
checkpoints/basic_gan_checkpoint.pth
checkpoints/dcgan_gan_checkpoint.pth
```

The AI Model Studio expects:

```text
checkpoints/face_generator.pth
```

The improved trainer writes `_v2` filenames. To use an improved checkpoint in
the current app, copy it to the filename expected by the app after confirming
that you want to replace the existing checkpoint:

```powershell
Copy-Item checkpoints\dcgan_gan_checkpoint_v2.pth `
  checkpoints\dcgan_gan_checkpoint.pth

Copy-Item checkpoints\face_generator_v2.pth `
  checkpoints\face_generator.pth
```

The improved Fashion checkpoint is compatible with the DCGAN generator
inference path. The improved face checkpoint is compatible with the existing
FaceGenerator inference path.

## Dataset details

### Fashion-MNIST

The project uses the Torchvision Fashion-MNIST dataset:

```text
Training images: 60,000
Test images: 10,000
Resolution: 28 x 28
Channels: 1 grayscale
Normalization: mean 0.5, standard deviation 0.5
Normalized range: [-1, 1]
```

### Face data

The original face loader accepts a user-provided recursive image folder and
resizes every image directly to 64x64. The improved loader uses a
center-square crop by default before resizing. No face dataset is bundled with
the repository.

## Project structure

```text
GAN-FashionVerse/
|-- app/                  Streamlit pages and shared UI
|-- data/                 Fashion-MNIST and face data loaders
|-- models/               Basic GAN, DCGAN, and face architectures
|-- training/             Original and improved training scripts
|-- inference/            Fashion and face inference
|-- datasets/             Downloaded datasets; generated locally
|-- checkpoints/          Model checkpoints; generated locally
|-- outputs/              Generated images, histories, and samples
|-- tests/                Automated tests
|-- config.py             Fashion defaults
|-- check_gpu.py          Runtime diagnostics
|-- requirements.txt      Dependencies
`-- README.md             This guide
```

## Important limitations

- Checkpoints are ignored by Git and are not guaranteed to be present in a
  fresh fork.
- Face dataset files are not included.
- The Fashion models operate at 28x28 grayscale resolution.
- The face models operate at 64x64 RGB resolution.
- No FID, Inception Score, or other image-quality metric is implemented.
- Training quality depends on the number of epochs, dataset, hardware, and
  selected trainer.
- Do not commit private datasets, credentials, or unlicensed image data.

## License and dataset responsibility

No project license file was found in the repository. Add an appropriate
license before redistributing the project. Verify the license and permitted
use of any face dataset before downloading, training, or distributing outputs.

## Author

GAN FashionVerse is an academic project by Harshvardhan1123.
