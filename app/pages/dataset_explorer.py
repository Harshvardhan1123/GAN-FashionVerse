from __future__ import annotations

import streamlit as st

from app.styles import hero
from data.dataset import (
    FASHION_MNIST_CLASSES,
    create_fashion_mnist_loader,
    dataset_metadata,
    plot_dataset_samples,
)


def render() -> None:
    hero("Dataset Explorer", "Inspect Fashion-MNIST before training.")
    download = st.checkbox("Download Fashion-MNIST if it is not present", value=False)
    split = st.selectbox("Dataset split", ["Training", "Test"])
    sample_count = st.slider("Preview samples", 4, 32, 16)
    if st.button("Load dataset", type="primary"):
        try:
            loader = create_fashion_mnist_loader(
                train=split == "Training",
                download=download,
                batch_size=32,
            )
            metadata = dataset_metadata(loader.dataset)
            st.session_state["dataset_metadata"] = metadata
            st.session_state["dataset_figure"] = plot_dataset_samples(
                loader.dataset, count=sample_count
            )
            st.session_state["dataset_batch"] = next(iter(loader))
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Dataset could not be loaded: {error}")
    metadata = st.session_state.get("dataset_metadata")
    if metadata:
        columns = st.columns(4)
        columns[0].metric("Images", metadata["samples"])
        columns[1].metric("Resolution", "28 × 28")
        columns[2].metric("Channels", metadata["channels"])
        columns[3].metric("Classes", metadata["classes"])
        st.json(metadata)
        figure = st.session_state.get("dataset_figure")
        if figure is not None:
            st.pyplot(figure)
        batch = st.session_state.get("dataset_batch")
        if batch is not None:
            st.write(f"Loaded batch shape: `{tuple(batch[0].shape)}`")
    st.caption("Classes: " + ", ".join(FASHION_MNIST_CLASSES))
