"""Tests for face cropper box helpers."""

import pytest
from PIL import Image

from minifigures_model.face_cropper import FaceBox, crop_face, normalized_box_to_pixel_box


def test_normalized_box_to_pixel_box_clamps_to_image() -> None:
    """Normalized boxes should resolve to valid PIL crop boxes."""
    box = FaceBox(x_center=0.1, y_center=0.1, width=0.4, height=0.4)

    assert normalized_box_to_pixel_box(box, (100, 80)) == (0, 0, 30, 24)


def test_face_box_rejects_invalid_width() -> None:
    """Face boxes must have positive normalized dimensions."""
    with pytest.raises(ValueError, match="width"):
        FaceBox(x_center=0.5, y_center=0.5, width=0, height=0.2)


def test_crop_face_returns_rgb_crop() -> None:
    """Face cropping should return a non-empty RGB image."""
    image = Image.new("RGB", (100, 100), color=(255, 255, 255))
    box = FaceBox(x_center=0.5, y_center=0.5, width=0.2, height=0.2)

    cropped = crop_face(image, box, padding=0)

    assert cropped.mode == "RGB"
    assert cropped.size == (20, 20)
