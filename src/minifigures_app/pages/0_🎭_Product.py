"""Streamlit image page."""
# ruff: noqa: INP001

from __future__ import annotations

from html import escape
from random import choice
from typing import NamedTuple

import streamlit as st
from PIL import Image

from minifigures_app.utils import get_image, get_prediction, list_im_tags, predict_image

HIGH_CONFIDENCE_THRESHOLD = 0.6
MEDIUM_CONFIDENCE_THRESHOLD = 0.4
MARKET_SELECTED_IMAGE_TAG = "market_selected_image_tag"


class ProductImage(NamedTuple):
    """Image selected for display with its optional catalog tag."""

    image: Image.Image
    tag: str | None


def main():
    """Run the Streamlit Product page."""
    st.set_page_config(page_title="Product", page_icon="🎭")

    # page header
    st.title("🎭 Product")
    st.markdown("---")

    # Toggle what to show
    options = ["Upload image", "Random example"]
    selected_product = st.session_state.get(MARKET_SELECTED_IMAGE_TAG)
    if selected_product:
        options = ["Marketplace selection", *options]

    show = st.radio("What do you want to do?", options)
    if show == "Marketplace selection":
        product_image = get_selected_from_marketplace()
    elif show == "Upload image":
        product_image = get_upload()
    elif show == "Random example":
        product_image = get_random()
    else:
        st.error("No option selected.")
        return

    # Show error if no image
    if product_image is None:
        st.error("No image selected.")
        return

    # Make prediction
    pred = predict_product_image(product_image)

    image_column, prediction_column = st.columns([1, 1.25], gap="large")
    with image_column:
        st.image(product_image.image, width="stretch")
    with prediction_column:
        if product_image.tag:
            st.caption(f"Prediction source: precomputed JSON for {product_image.tag}")
        st.write("Predictions:")
        render_prediction_bars(pred)


def predict_product_image(product_image: ProductImage) -> dict[str, float]:
    """Predict using precomputed catalog scores when a tag is available."""
    if product_image.tag:
        return get_prediction(product_image.tag)
    return predict_image(image=product_image.image)


def render_prediction_bars(predictions: dict[str, float]) -> None:
    """Render predictions as percentage bars."""
    rows = "\n".join(
        _prediction_bar_row(label=label, value=value)
        for label, value in sorted(predictions.items(), key=lambda item: item[0])
    )
    st.markdown(
        f"""
<style>
.prediction-bars {{
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    margin-top: 0.45rem;
}}
.prediction-row {{
    display: grid;
    grid-template-columns: minmax(8.5rem, 38%) 1fr;
    align-items: center;
    gap: 0.8rem;
}}
.prediction-label {{
    color: #2f4858;
    font-family: monospace;
    font-size: 0.95rem;
    overflow-wrap: anywhere;
}}
.prediction-track {{
    background: #edf2f2;
    border-radius: 999px;
    height: 1.85rem;
    overflow: hidden;
    position: relative;
}}
.prediction-fill {{
    align-items: center;
    border-radius: 999px;
    color: #ffffff;
    display: flex;
    font-size: 0.82rem;
    font-weight: 600;
    height: 100%;
}}
.prediction-fill.high {{
    background: #2fb872;
}}
.prediction-fill.medium {{
    background: #f0c14b;
    color: #2b2b2b;
}}
.prediction-fill.low {{
    background: #ef7f7f;
}}
.prediction-percent {{
    color: #2b2b2b;
    font-size: 0.82rem;
    font-weight: 600;
    left: 0.65rem;
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    z-index: 1;
}}
</style>
<div class="prediction-bars">
{rows}
</div>
""",
        unsafe_allow_html=True,
    )


def _prediction_bar_row(label: str, value: float) -> str:
    normalized_value = min(max(float(value), 0), 1)
    percentage = round(normalized_value * 100)
    color_class = _prediction_color_class(normalized_value)
    return (
        '<div class="prediction-row">'
        f'<div class="prediction-label">"{escape(label)}":</div>'
        '<div class="prediction-track">'
        f'<span class="prediction-percent">{percentage}%</span>'
        f'<div class="prediction-fill {color_class}" style="width: {percentage}%;"></div>'
        "</div>"
        "</div>"
    )


def _prediction_color_class(value: float) -> str:
    if value >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if value >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def get_upload() -> ProductImage | None:
    """Get an image from the user."""
    # Upload a file
    uploaded_file = st.file_uploader("Upload image", ["png", "jpg"], accept_multiple_files=False)

    # Create prediction for the file
    if uploaded_file:
        # Convert to PIL
        return ProductImage(image=Image.open(uploaded_file).convert("RGB"), tag=None)
    return None


def get_random() -> ProductImage | None:
    """Get a random image."""
    # Get all possible image tags
    im_tags = list_im_tags()

    # Randomly select one and return it
    image_tag = choice(im_tags)  # noqa: S311
    return ProductImage(image=get_image(image_tag), tag=image_tag)


def get_selected_from_marketplace() -> ProductImage | None:
    """Get the image selected from marketplace session state."""
    image_tag = st.session_state.get(MARKET_SELECTED_IMAGE_TAG)
    if not image_tag:
        return None
    st.caption(f"Selected from market: {image_tag}")
    return ProductImage(image=get_image(image_tag), tag=image_tag)


if __name__ == "__main__":
    main()
