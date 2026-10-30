from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.styles import empty_state, hero, section_heading
from inference.generate_fashion import generate_fashion_images, save_fashion_images


def render() -> None:
    hero("Fashion Studio", "Turn a latent seed into a new fashion concept.", "Creative workspace")
    st.caption("Output is 28×28 grayscale Fashion-MNIST imagery, not photorealistic full-body fashion.")
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        st.markdown('<div class="studio-panel">', unsafe_allow_html=True)
        architecture = st.selectbox("Architecture", ["basic", "dcgan"])
        count = st.slider("Design count", min_value=1, max_value=16, value=4)
        seed = st.number_input("Seed", min_value=0, value=42, help="Reuse a seed to reproduce the same latent vectors.")
        generate = st.button("Generate designs", type="primary", width="stretch")
        st.caption("Different seeds produce different latent vectors and therefore different outputs.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        section_heading("Latest collection", "Your generated designs appear here after a successful run.")
    checkpoint = Path(f"checkpoints/{architecture}_gan_checkpoint.pth")
    if not checkpoint.exists():
        empty_state("Checkpoint required", f"Train the {architecture.upper()} model before generating designs.")
        return

    if generate:
        try:
            images = generate_fashion_images(
                checkpoint,
                count=count,
                seed=int(seed),
                architecture=architecture,
            )
            paths = save_fashion_images(images, seed=int(seed))
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            st.error(str(error))
            return
        st.session_state["latest_fashion_images"] = images
        st.session_state["latest_fashion_paths"] = paths

    images = st.session_state.get("latest_fashion_images")
    paths = st.session_state.get("latest_fashion_paths", [])
    if images is not None:
        display_paths = [str(path) for path in paths if Path(path).is_file()]
        if display_paths:
            st.image(display_paths, width=180)
            download_columns = st.columns(min(len(paths), 4))
            for index, path in enumerate(paths):
                with download_columns[index % len(download_columns)]:
                    st.download_button(
                        f"Download {index + 1:02d}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="image/png",
                        key=str(path),
                        width="stretch",
                    )
