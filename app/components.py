"""Reusable Streamlit components."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_sidebar() -> str:
    """Render navigation and return the selected page."""
    with st.sidebar:
        st.markdown(
            '<div class="brand-mark"><div class="brand-orb"></div>'
            '<div class="brand-name">GAN FashionVerse</div></div>'
            '<div class="brand-tagline">Where artificial intelligence meets fashion creativity.</div>',
            unsafe_allow_html=True,
        )
        return st.radio(
            "Navigate",
            [
                "Home",
                "Fashion Studio",
                "AI Model Studio",
                "Training Dashboard",
                "Dataset Explorer",
                "How GAN Works",
                "About Project",
            ],
            label_visibility="collapsed",
            key="main_navigation",
        )


def checkpoint_status(path: str | Path) -> str:
    """Return a truthful user-facing model status."""
    return "Trained checkpoint available" if Path(path).is_file() else "Not trained yet"


def count_gallery_images(path: str | Path) -> int:
    """Count persisted gallery records without failing on a missing gallery."""
    import json

    gallery_path = Path(path)
    if not gallery_path.is_file():
        return 0
    try:
        records = json.loads(gallery_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    return len(records) if isinstance(records, list) else 0
