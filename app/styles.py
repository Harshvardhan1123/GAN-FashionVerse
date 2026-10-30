"""Visual styling for the GAN FashionVerse Streamlit application."""

from __future__ import annotations

import streamlit as st


def apply_styles() -> None:
    """Apply the shared dark fashion-tech theme."""
    st.markdown(
        """
        <style>
        :root {
            --fv-bg: #09090d;
            --fv-surface: #12121a;
            --fv-surface-2: #181823;
            --fv-border: rgba(255,255,255,.09);
            --fv-muted: #9e9baa;
            --fv-text: #f7f5fb;
            --fv-pink: #e9a7ff;
            --fv-purple: #8d68ff;
        }
        .stApp {
            background: var(--fv-bg);
            color: var(--fv-text);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        [data-testid="stSidebar"] {
            background: #0f0f16;
            border-right: 1px solid var(--fv-border);
        }
        [data-testid="stSidebar"] > div:first-child {
            padding: 1.4rem 1rem;
        }
        [data-testid="stSidebar"] [role="radiogroup"] {
            gap: .35rem;
        }
        [data-testid="stSidebar"] label {
            border-radius: .7rem;
            padding: .35rem .55rem;
            transition: background .2s ease;
        }
        [data-testid="stSidebar"] label:hover {
            background: rgba(233,167,255,.08);
        }
        h1, h2, h3 {
            letter-spacing: -.03em;
        }
        .brand-mark {
            display: flex;
            align-items: center;
            gap: .65rem;
            margin-bottom: .35rem;
        }
        .brand-orb {
            width: 2rem;
            height: 2rem;
            border-radius: .7rem;
            background: linear-gradient(135deg, var(--fv-pink), var(--fv-purple));
            box-shadow: 0 0 24px rgba(141,104,255,.35);
        }
        .brand-name {
            color: var(--fv-text);
            font-size: 1.05rem;
            font-weight: 750;
        }
        .brand-tagline {
            color: var(--fv-muted);
            font-size: .78rem;
            line-height: 1.35;
            margin-bottom: 1.25rem;
        }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.4rem 2.5rem;
            border: 1px solid rgba(233,167,255,.2);
            border-radius: 1.5rem;
            background:
                radial-gradient(circle at 88% 18%, rgba(233,167,255,.2), transparent 30%),
                linear-gradient(135deg, #171522 0%, #20162d 58%, #321a3d 100%);
            margin-bottom: 1.8rem;
            box-shadow: 0 18px 70px rgba(0,0,0,.2);
        }
        .hero::after {
            content: "";
            position: absolute;
            width: 16rem;
            height: 16rem;
            right: -5rem;
            bottom: -8rem;
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 50%;
        }
        .hero-kicker {
            color: var(--fv-pink);
            font-size: .74rem;
            font-weight: 750;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin-bottom: .65rem;
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(2.1rem, 5vw, 3.6rem);
            background: linear-gradient(90deg, #ffffff, #e1a7ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero p {
            max-width: 42rem;
            color: #c9c4d5;
            font-size: 1.02rem;
            line-height: 1.6;
            margin-bottom: 0;
        }
        .metric-card {
            padding: 1.15rem 1.25rem;
            border: 1px solid var(--fv-border);
            border-radius: 1rem;
            background: linear-gradient(145deg, var(--fv-surface-2), var(--fv-surface));
            min-height: 6.4rem;
            box-shadow: 0 10px 30px rgba(0,0,0,.12);
        }
        .metric-label {
            color: var(--fv-muted);
            font-size: .76rem;
            letter-spacing: .05em;
            text-transform: uppercase;
        }
        .metric-value {
            color: var(--fv-text);
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: .5rem;
        }
        .section-heading {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 1rem;
            margin: 2rem 0 .85rem;
        }
        .section-heading h2 {
            margin: 0;
            font-size: 1.35rem;
        }
        .section-heading p {
            color: var(--fv-muted);
            margin: 0;
            font-size: .9rem;
        }
        .feature-card {
            min-height: 10rem;
            padding: 1.25rem;
            border: 1px solid var(--fv-border);
            border-radius: 1rem;
            background: var(--fv-surface);
        }
        .feature-icon {
            color: var(--fv-pink);
            font-size: 1.25rem;
            margin-bottom: .7rem;
        }
        .feature-card h3 {
            font-size: 1.05rem;
            margin: 0 0 .45rem;
        }
        .feature-card p {
            color: var(--fv-muted);
            font-size: .88rem;
            line-height: 1.5;
            margin: 0;
        }
        .eyebrow {
            color: var(--fv-pink);
            font-size: .75rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
        .studio-panel {
            padding: 1.3rem;
            border: 1px solid var(--fv-border);
            border-radius: 1rem;
            background: var(--fv-surface);
        }
        .empty-state {
            padding: 2.5rem 1.5rem;
            text-align: center;
            border: 1px dashed rgba(233,167,255,.25);
            border-radius: 1rem;
            background: rgba(233,167,255,.035);
        }
        .empty-state h3 {
            margin: .4rem 0;
        }
        .empty-state p {
            color: var(--fv-muted);
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, kicker: str = "GAN FashionVerse") -> None:
    """Render a shared page hero."""
    st.markdown(
        f'<section class="hero"><div class="hero-kicker">{kicker}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></section>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> None:
    """Render a compact metric card."""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def section_heading(title: str, description: str = "") -> None:
    """Render a consistent section title and supporting copy."""
    st.markdown(
        f'<div class="section-heading"><div><h2>{title}</h2>'
        f'<p>{description}</p></div></div>',
        unsafe_allow_html=True,
    )


def feature_card(icon: str, title: str, description: str) -> None:
    """Render a compact feature card."""
    st.markdown(
        f'<div class="feature-card"><div class="feature-icon">{icon}</div>'
        f'<h3>{title}</h3><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, description: str) -> None:
    """Render a calm empty state for unavailable artifacts."""
    st.markdown(
        f'<div class="empty-state"><div class="feature-icon">✦</div>'
        f'<h3>{title}</h3><p>{description}</p></div>',
        unsafe_allow_html=True,
    )
