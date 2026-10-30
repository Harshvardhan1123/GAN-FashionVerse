"""Streamlit entry point for GAN FashionVerse."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit may execute this file directly, which otherwise shadows the app package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components import render_sidebar
from app.pages import about, dataset_explorer, fashion_studio, gan_explainer, home, model_studio, training_dashboard
from app.styles import apply_styles


def main() -> None:
    """Render the selected application page."""
    st.set_page_config(
        page_title="GAN FashionVerse",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()
    page = render_sidebar()
    renderers = {
        "Home": home.render,
        "Fashion Studio": fashion_studio.render,
        "AI Model Studio": model_studio.render,
        "Training Dashboard": training_dashboard.render,
        "Dataset Explorer": dataset_explorer.render,
        "How GAN Works": gan_explainer.render,
        "About Project": about.render,
    }
    renderers[page]()


main()
