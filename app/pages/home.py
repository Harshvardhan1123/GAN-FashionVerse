from __future__ import annotations

from pathlib import Path

import streamlit as st

import json

from app.components import checkpoint_status, count_gallery_images
from app.styles import feature_card, hero, metric_card, section_heading


def render() -> None:
    hero("Creative intelligence for fashion.", "Train, explore, and generate with real PyTorch GAN models.")
    checkpoint = Path("checkpoints/basic_gan_checkpoint.pth")
    gallery = Path("outputs/fashion/gallery.json")
    columns = st.columns(3)
    with columns[0]:
        metric_card("Generated images", str(count_gallery_images(gallery)))
    with columns[1]:
        metric_card("Active architecture", "Basic GAN")
    with columns[2]:
        metric_card("Training status", checkpoint_status(checkpoint))

    section_heading("Your creative workspace", "Everything you need to move from experiment to generated concept.")
    cards = st.columns(3)
    with cards[0]:
        feature_card("✦", "Fashion Studio", "Generate 28×28 grayscale Fashion-MNIST designs from a trained GAN.")
    with cards[1]:
        feature_card("◌", "Training Dashboard", "Inspect real generator and discriminator losses from your experiments.")
    with cards[2]:
        feature_card("↗", "How GAN Works", "Understand how noise, adversarial loss, and two competing networks create images.")

    recent_paths = []
    if gallery.exists():
        try:
            records = json.loads(gallery.read_text(encoding="utf-8"))
            recent_paths = [
                record["path"]
                for record in records[-4:]
                if isinstance(record, dict) and Path(record.get("path", "")).is_file()
            ]
        except (OSError, json.JSONDecodeError, TypeError):
            recent_paths = []
    if recent_paths:
        section_heading("Recent designs", "Saved outputs from your latest generation runs.")
        st.image(recent_paths, width=170)
    if not checkpoint.exists():
        st.warning(
            "No trained fashion checkpoint is available yet. Train the Basic GAN before "
            "using image generation; the interface will not display placeholder images."
        )
