"""Streamlit helper to manually annotate minifigure face boxes.

Run from the repository root with:

    export PYTHONPATH=src
    streamlit run src/labeling/annotate_face_boxes.py

Adjust the sliders until the red rectangle and the crop preview line up with
the face, then click "Save & next". Progress is written to face_boxes.json
after every save, so the session can be closed and resumed at any time.
"""
# ruff: noqa: INP001

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from minifigures_model.data_utils import list_image_paths, load_json, write_json
from minifigures_model.face_cropper import DEFAULT_FACE_CROPPER_RESOLUTION, FaceBox, crop_face
from minifigures_model.face_cropper_train import DEFAULT_FACE_BOXES_PATH

DEFAULT_BOX = {"x_center": 0.5, "y_center": 0.25, "width": 0.3, "height": 0.3}
SLIDER_STEP = 0.01
OVERLAY_COLOR = "red"
OVERLAY_WIDTH = 3


def _load_face_boxes(path: Path) -> dict[str, dict[str, float]]:
    """Load existing face box annotations, if any."""
    if not path.is_file():
        return {}
    return load_json(path)


def _draw_box_overlay(image: Image.Image, box: dict[str, float]) -> Image.Image:
    """Draw the candidate face box on top of a copy of the image."""
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    left = (box["x_center"] - box["width"] / 2) * width
    top = (box["y_center"] - box["height"] / 2) * height
    right = (box["x_center"] + box["width"] / 2) * width
    bottom = (box["y_center"] + box["height"] / 2) * height
    draw.rectangle([left, top, right, bottom], outline=OVERLAY_COLOR, width=OVERLAY_WIDTH)
    return overlay


def main() -> None:
    """Run the Streamlit face box annotation helper."""
    st.set_page_config(page_title="Annotate Face Boxes", layout="wide")
    st.title("Annotate Face Boxes")
    st.caption(f"Writing annotations to {DEFAULT_FACE_BOXES_PATH}")

    image_paths = list_image_paths()
    face_boxes = _load_face_boxes(DEFAULT_FACE_BOXES_PATH)

    if "annotation_queue" not in st.session_state:
        st.session_state["annotation_queue"] = [
            path for path in image_paths if path.stem not in face_boxes
        ]
        st.session_state["annotation_index"] = 0

    queue = st.session_state["annotation_queue"]
    index = st.session_state["annotation_index"]

    st.write(f"Annotated so far: **{len(face_boxes)} / {len(image_paths)}**")

    if index >= len(queue):
        st.success("No more images left in this session's queue.")
        st.info("Restart the app to pick up any images annotated in other sessions.")
        return

    image_path = queue[index]
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")

    st.subheader(f"{image_path.stem}  —  {index + 1} / {len(queue)} remaining")

    slider_prefix = image_path.stem
    box = {
        "x_center": st.slider(
            "x_center", 0.0, 1.0, DEFAULT_BOX["x_center"], SLIDER_STEP, key=f"{slider_prefix}_x"
        ),
        "y_center": st.slider(
            "y_center", 0.0, 1.0, DEFAULT_BOX["y_center"], SLIDER_STEP, key=f"{slider_prefix}_y"
        ),
        "width": st.slider(
            "width", 0.01, 1.0, DEFAULT_BOX["width"], SLIDER_STEP, key=f"{slider_prefix}_w"
        ),
        "height": st.slider(
            "height", 0.01, 1.0, DEFAULT_BOX["height"], SLIDER_STEP, key=f"{slider_prefix}_h"
        ),
    }

    preview_column, crop_column = st.columns(2)
    with preview_column:
        st.image(_draw_box_overlay(image, box), caption="Box preview", width="stretch")

    face_box = None
    with crop_column:
        try:
            face_box = FaceBox(**box)
            cropped = crop_face(image, face_box).resize(
                (DEFAULT_FACE_CROPPER_RESOLUTION, DEFAULT_FACE_CROPPER_RESOLUTION)
            )
            st.image(cropped, caption="Cropped face preview", width="stretch")
        except ValueError as exc:
            st.warning(f"Invalid box: {exc}")

    save_column, skip_column, back_column = st.columns(3)
    if save_column.button("Save & next", disabled=face_box is None, type="primary"):
        face_boxes[image_path.stem] = face_box.to_dict()
        write_json(DEFAULT_FACE_BOXES_PATH, face_boxes, sort_keys=True)
        st.session_state["annotation_index"] += 1
        st.rerun()
    if skip_column.button("Skip"):
        st.session_state["annotation_index"] += 1
        st.rerun()
    if back_column.button("Back", disabled=index == 0):
        st.session_state["annotation_index"] -= 1
        st.rerun()


if __name__ == "__main__":
    main()
