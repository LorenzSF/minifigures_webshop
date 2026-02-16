"""Streamlit image page."""

import streamlit as st
from PIL import Image

from minifigures_app.utils import predict_image


def main():
    """Main function."""
    st.set_page_config(page_title="Product", page_icon="🎭")

    # page header
    st.title("🎭 Product")
    st.markdown("---")

    # Upload a file
    uploaded_file = st.file_uploader("Upload image", ["png", "jpg"], accept_multiple_files=False)

    # Create prediction for the file
    if uploaded_file:
        # Convert to PIL
        im = Image.open(uploaded_file).convert("RGB")

        # Make prediction
        pred = predict_image(image=im)

        # Show result
        st.write("Predictions:")
        st.write(pred)


if __name__ == "__main__":
    main()
