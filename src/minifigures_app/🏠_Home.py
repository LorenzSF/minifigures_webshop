"""Streamlit app."""

import base64
from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).parent / "assets"
HERO_IMAGE_PATH = ASSETS_DIR / "minifigure_hero.png"
HERO_FRAME_WIDTH_PX = 380

HERO_STYLE = f"""
<style>
.hero-row {{
    display: flex;
    align-items: stretch;
    gap: 1.5rem;
    margin-bottom: 1rem;
}}
.hero-image-frame {{
    flex: 0 0 {HERO_FRAME_WIDTH_PX}px;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.hero-image-frame img {{
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}}
.speech-bubble {{
    position: relative;
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    background: #ffffff;
    border: 2px solid #1a1a2e;
    border-radius: 18px;
    padding: 1.5rem 1.75rem;
    margin-left: 1.5rem;
}}
.speech-bubble::before {{
    content: "";
    position: absolute;
    top: 50%;
    left: -22px;
    transform: translateY(-50%);
    border-style: solid;
    border-width: 14px 22px 14px 0;
    border-color: transparent #1a1a2e transparent transparent;
}}
.speech-bubble::after {{
    content: "";
    position: absolute;
    top: 50%;
    left: -18px;
    transform: translateY(-50%);
    border-style: solid;
    border-width: 11px 18px 11px 0;
    border-color: transparent #ffffff transparent transparent;
}}
.speech-bubble h1 {{
    margin-top: 0;
}}
</style>
"""


def _image_data_uri(path: Path) -> str:
    """Return a base64 data URI so the image can be embedded in raw HTML."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def main():
    """Main function."""
    # main page config
    st.set_page_config(page_title="Minifigures Webshop", page_icon="🏠")
    st.markdown(HERO_STYLE, unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="hero-row">
<div class="hero-image-frame">
<img src="{_image_data_uri(HERO_IMAGE_PATH)}" alt="Minifigure mascot" />
</div>
<div class="speech-bubble">
<h1>Welcome to the Lorenzo's Minifigures webshop!</h1>
<p>Here you'll find more than 6,000 images to choose from, but only one will be your perfect clone. Let's find it!</p>
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 👈 Pages")
    st.markdown(" - **🎭 Product**: Upload and view your product")
    st.markdown(" - **💰 Market**: Go to our marketplace")
    st.markdown(
        " - **🔍 Find your LEGO clone**: Search by picture or webcam photo, your similar pieces"
    )


if __name__ == "__main__":
    main()
