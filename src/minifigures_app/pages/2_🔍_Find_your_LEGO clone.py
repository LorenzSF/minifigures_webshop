"""Streamlit face similarity search page."""
# ruff: noqa: INP001

from __future__ import annotations

import streamlit as st
from PIL import Image, ImageOps

from minifigures_app.utils import get_image, search_similar_faces

DEFAULT_TOP_K = 6
MAX_TOP_K = 24
RESULT_IMAGE_SIZE = 180


def main() -> None:
    """Run the Streamlit Face Search page."""
    st.set_page_config(page_title="Face Search")

    st.title("Face Search")
    st.markdown("---")

    source = st.radio("Image source", ["Upload a file", "Use webcam"], horizontal=True)
    if source == "Upload a file":
        query_file = st.file_uploader("Upload a face or minifigure image", ["png", "jpg", "jpeg"])
    else:
        query_file = st.camera_input("Take a picture")

    top_k = st.slider("Results", min_value=1, max_value=MAX_TOP_K, value=DEFAULT_TOP_K)

    if query_file is None:
        return

    query_image = Image.open(query_file).convert("RGB")
    st.image(query_image, caption="Query image", width=260)

    try:
        payload = search_similar_faces(query_image, top_k=top_k)
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        return

    matches = payload["matches"]
    if not matches:
        st.info("No face matches found.")
        return

    st.subheader("Similar faces")
    columns = st.columns(3)
    for idx, match in enumerate(matches):
        column = columns[idx % len(columns)]
        with column:
            image = get_image(match["tag"])
            st.image(_rescale_image(image), width="stretch")
            st.write(match["tag"])
            st.caption(f"Similarity: {match['score']:.2f}")


def _rescale_image(image: Image.Image) -> Image.Image:
    """Rescale a marketplace image for result display."""
    size = (RESULT_IMAGE_SIZE, RESULT_IMAGE_SIZE)
    return ImageOps.pad(ImageOps.contain(image, size=size), size=size, color="white")


if __name__ == "__main__":
    main()
