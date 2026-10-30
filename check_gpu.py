"""Report the active Python, PyTorch, and CUDA environment."""

from __future__ import annotations

import logging
import platform
import sys

import torch


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Print runtime and device information for training diagnostics."""
    cuda_available = torch.cuda.is_available()
    LOGGER.info("Python: %s", sys.version.split()[0])
    LOGGER.info("Platform: %s", platform.platform())
    LOGGER.info("PyTorch: %s", torch.__version__)
    LOGGER.info("CUDA available: %s", cuda_available)
    if cuda_available:
        LOGGER.info("CUDA version: %s", torch.version.cuda)
        LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        LOGGER.info("Using CPU fallback")


if __name__ == "__main__":
    main()
