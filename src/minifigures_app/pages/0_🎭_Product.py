"""Streamlit image page."""

from __future__ import annotations

from random import choice

import streamlit as st
from PIL import Image

from minifigures_app.utils import get_image, list_im_tags, predict_image


def main():
    """Main function."""
    st.set_page_config(page_title="Product", page_icon="🎭")

    # page header
    st.title("🎭 Product")
    st.markdown("---")

    # Toggle what to show
    options = ["Upload image", "Random example"]
    selected_product = st.session_state.get("selected_product_tag")
    if selected_product:
        options = ["Marketplace selection", *options]

    show = st.radio("What do you want to do?", options)
    if show == "Marketplace selection":
        im = get_selected_from_marketplace()
    elif show == "Upload image":
        im = get_upload()
    elif show == "Random example":
        im = get_random()
    else:
        st.error("No option selected.")
        return

    # Show error if no image
    if im is None:
        st.error("No image selected.")
        return

    # Show the image
    st.image(im)

    # Make prediction
    pred = predict_image(image=im)

    # Show result
    st.write("Predictions:")
    st.write(pred)


def get_upload() -> Image.Image | None:
    """Get an image from the user."""
    # Upload a file
    uploaded_file = st.file_uploader("Upload image", ["png", "jpg"], accept_multiple_files=False)

    # Create prediction for the file
    if uploaded_file:
        # Convert to PIL
        return Image.open(uploaded_file).convert("RGB")
    return None


def get_random() -> Image.Image | None:
    """Get a random image."""
    # Get all possible image tags
    im_tags = list_im_tags()

    # Randomly select one and return it
    im = choice(im_tags)
    return get_image(im)


def get_selected_from_marketplace() -> Image.Image | None:
    """Get the image selected from marketplace session state."""
    image_tag = st.session_state.get("selected_product_tag")
    if not image_tag:
        return None
    st.caption(f"Selected from market: {image_tag}")
    return get_image(image_tag)


if __name__ == "__main__":
    main()
