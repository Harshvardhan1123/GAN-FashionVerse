from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch

from app.styles import hero, section_heading


def render() -> None:
    hero("About Project", "A transparent workspace for learning and experimenting with GANs.", "Project notes")
    st.markdown(
        """
        GAN FashionVerse demonstrates practical GAN implementation with PyTorch,
        Torchvision, Fashion-MNIST, and Streamlit. The project keeps data loading,
        architectures, training, inference, and UI logic modular.

        **Current scope**
        - Basic fully connected GAN
        - DCGAN architecture
        - Actual Fashion-MNIST training and inference
        - Checkpoint, loss-history, and generated-image persistence

        **Limitations**
        - Fashion output is 28×28 grayscale because it is trained on Fashion-MNIST.
        - Face generation is not available until a dedicated face model is trained.
        """
    )
    section_heading("Runtime diagnostics", "Facts from the environment running this app.")
    columns = st.columns(4)
    columns[0].metric("PyTorch", torch.__version__)
    columns[1].metric("Device", "CUDA" if torch.cuda.is_available() else "CPU")
    columns[2].metric("Fashion model", "Basic + DCGAN")
    columns[3].metric("Face model", "FFHQ-trained")
    st.download_button(
        "Download README",
        data=Path("README.md").read_bytes(),
        file_name="GAN-FashionVerse-README.md",
        mime="text/markdown",
        width="content",
    )
