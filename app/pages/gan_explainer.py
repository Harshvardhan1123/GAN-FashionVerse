from __future__ import annotations

import streamlit as st

from app.styles import hero


def render() -> None:
    hero("How GAN Works", "Learn the loop behind every generated concept.", "Learning lab")
    topic = st.radio("Choose a topic", ["Overview", "Generator", "Discriminator", "Training loop"], horizontal=True)
    if topic == "Overview":
        st.markdown(
        """
        ### What is a GAN?
        A Generative Adversarial Network trains two models together. The **Generator**
        creates candidates from random noise, while the **Discriminator** predicts
        whether an image is real or generated.

        ```text
        Random Noise ──> GENERATOR ──> Fake Fashion Image ──> DISCRIMINATOR ──> Real/Fake
        Real Fashion Image ────────────────────────────────> DISCRIMINATOR
        ```

        The discriminator learns from real and detached fake images. The generator then
        learns to make the discriminator classify newly generated images as real.
        Both models use binary cross-entropy with logits, and the losses recorded in the
        Training Dashboard come from the actual training loop.
        """
        )
    elif topic == "Generator":
        st.subheader("The generator")
        st.write("The generator maps a 100-dimensional random noise vector to a normalized image tensor. It learns by receiving gradients from the discriminator when its images are classified as fake.")
        st.code("noise (100) → linear layers → 784 values → reshape (1, 28, 28)", language="text")
    elif topic == "Discriminator":
        st.subheader("The discriminator")
        st.write("The discriminator receives real or generated images and returns a raw logit. BCEWithLogitsLoss turns that logit into the adversarial classification objective.")
        st.code("image (1, 28, 28) → flatten → linear layers → real/fake logit", language="text")
    else:
        st.subheader("One training step")
        st.write("1. Update the discriminator with real images and detached fake images. 2. Generate a fresh batch. 3. Update the generator against real targets. 4. Record both losses.")
        st.progress(0.75, text="Discriminator → Generator → Metrics")
