from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from app.styles import hero


def render() -> None:
    hero("Training Dashboard", "Review metrics and samples produced by actual GAN training runs.", "Experiment monitor")
    architecture = st.selectbox("Architecture history", ["basic", "dcgan"])
    history_path = Path(f"outputs/training_graphs/{architecture}_gan_history.json")
    if not history_path.exists():
        st.info(f"No {architecture.upper()} training history is available yet.")
        return
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read training history: {error}")
        return

    generator_loss = history.get("generator_loss", [])
    discriminator_loss = history.get("discriminator_loss", [])
    if not isinstance(generator_loss, list) or not isinstance(discriminator_loss, list):
        st.error("Training history has an invalid format.")
        return
    figure, axis = plt.subplots()
    figure.patch.set_facecolor("#12121a")
    axis.set_facecolor("#12121a")
    axis.plot(generator_loss, label="Generator loss")
    axis.plot(discriminator_loss, label="Discriminator loss")
    axis.set_xlabel("Epoch", color="#c9c4d5")
    axis.set_ylabel("BCE-with-logits loss", color="#c9c4d5")
    axis.tick_params(colors="#c9c4d5")
    axis.legend()
    axis.grid(alpha=0.2)
    st.pyplot(figure)
    plt.close(figure)
    st.metric("Completed epochs", len(generator_loss))

    samples = sorted(Path("outputs/training_graphs").glob(f"{architecture}_gan_epoch_*.png"))
    if samples:
        st.subheader(f"{architecture.upper()} samples by epoch")
        st.image([str(path) for path in samples], width=220)
    checkpoint = Path(f"checkpoints/{architecture}_gan_checkpoint.pth")
    st.caption(f"Checkpoint: `{checkpoint}` — {'available' if checkpoint.exists() else 'missing'}")
