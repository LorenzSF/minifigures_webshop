"""Face cropper model and box utilities for minifigure face search."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torchvision import models

from minifigures_model.constants import get_models_folder
from minifigures_model.preprocessing import preprocess_pil_image

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image

DEFAULT_FACE_CROPPER_TAG = "face_cropper"
DEFAULT_FACE_CROPPER_RESOLUTION = 256
FACE_BOX_KEYS = ("x_center", "y_center", "width", "height")


@dataclass(frozen=True)
class FaceBox:
    """Normalized center-format face box."""

    x_center: float
    y_center: float
    width: float
    height: float

    def __post_init__(self) -> None:
        """Validate values after dataclass initialization."""
        validate_face_box(self)

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable representation."""
        return {
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height,
        }

    def to_tensor(self) -> torch.Tensor:
        """Return the box as a float tensor."""
        return torch.tensor(
            [self.x_center, self.y_center, self.width, self.height], dtype=torch.float32
        )

    @classmethod
    def from_tensor(cls, box: torch.Tensor) -> FaceBox:
        """Create a face box from a four-value tensor."""
        if box.numel() != len(FACE_BOX_KEYS):
            msg = f"Expected {len(FACE_BOX_KEYS)} box values, got {box.numel()}."
            raise ValueError(msg)
        values = box.detach().cpu().flatten().tolist()
        return cls(
            x_center=float(values[0]),
            y_center=float(values[1]),
            width=float(values[2]),
            height=float(values[3]),
        )

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> FaceBox:
        """Create a face box from a JSON-like mapping."""
        missing_keys = [key for key in FACE_BOX_KEYS if key not in payload]
        if missing_keys:
            msg = f"Face box is missing keys: {missing_keys}"
            raise ValueError(msg)
        return cls(**{key: float(payload[key]) for key in FACE_BOX_KEYS})


def validate_face_box(box: FaceBox) -> None:
    """Validate normalized center-format face box values."""
    values = box.to_dict()
    non_finite = [key for key, value in values.items() if not math.isfinite(value)]
    if non_finite:
        msg = f"Face box contains non-finite values: {non_finite}"
        raise ValueError(msg)

    if not 0 <= box.x_center <= 1:
        msg = f"x_center must be within [0, 1], got {box.x_center}"
        raise ValueError(msg)
    if not 0 <= box.y_center <= 1:
        msg = f"y_center must be within [0, 1], got {box.y_center}"
        raise ValueError(msg)
    if not 0 < box.width <= 1:
        msg = f"width must be within (0, 1], got {box.width}"
        raise ValueError(msg)
    if not 0 < box.height <= 1:
        msg = f"height must be within (0, 1], got {box.height}"
        raise ValueError(msg)


def normalized_box_to_pixel_box(
    box: FaceBox, image_size: tuple[int, int], *, padding: float = 0.0
) -> tuple[int, int, int, int]:
    """Convert a normalized center-format box to a clamped PIL pixel box."""
    if padding < 0:
        msg = f"padding must be non-negative, got {padding}"
        raise ValueError(msg)

    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        msg = f"Invalid image size: {image_size}"
        raise ValueError(msg)

    padded_width = min(box.width * (1 + padding), 1.0)
    padded_height = min(box.height * (1 + padding), 1.0)
    left = round((box.x_center - padded_width / 2) * image_width)
    top = round((box.y_center - padded_height / 2) * image_height)
    right = round((box.x_center + padded_width / 2) * image_width)
    bottom = round((box.y_center + padded_height / 2) * image_height)

    left = max(left, 0)
    top = max(top, 0)
    right = min(right, image_width)
    bottom = min(bottom, image_height)

    if right <= left or bottom <= top:
        msg = f"Face box resolves to an empty crop: {(left, top, right, bottom)}"
        raise ValueError(msg)

    return left, top, right, bottom


def crop_face(image: Image.Image, box: FaceBox, *, padding: float = 0.15) -> Image.Image:
    """Crop a face from an image using a normalized face box."""
    pixel_box = normalized_box_to_pixel_box(box, image.size, padding=padding)
    return image.convert("RGB").crop(pixel_box)


class FaceCropper(nn.Module):
    """Predict a normalized face box for a minifigure image."""

    def __init__(
        self,
        tag: str = DEFAULT_FACE_CROPPER_TAG,
        resolution: int = DEFAULT_FACE_CROPPER_RESOLUTION,
        *,
        pretrained: bool = False,
        freeze_encoder: bool = True,
    ) -> None:
        """Initialize the cropper model."""
        super().__init__()
        self.tag = tag
        self.resolution = resolution

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.encoder = models.efficientnet_b0(weights=weights)
        n_features = self.encoder.classifier[1].in_features
        self.encoder.classifier[-1] = torch.nn.Identity()
        self.decoder = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Linear(128, len(FACE_BOX_KEYS)), nn.Sigmoid()
        )

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict normalized box values for a batch of images."""
        features = self.encoder(image)
        return self.decoder(features)

    def predict_box(self, image: Image.Image) -> FaceBox:
        """Predict a normalized face box for a PIL image."""
        self.eval()
        with torch.inference_mode():
            image_t = preprocess_pil_image(image, resolution=self.resolution)
            box_t = self.forward(image_t[None,])[0]
        return FaceBox.from_tensor(box_t)

    def get_metadata(self) -> dict[str, Any]:
        """Return model metadata."""
        return {"tag": self.tag, "resolution": self.resolution}

    def save(self) -> None:
        """Save cropper metadata and weights."""
        model_folder = get_models_folder() / self.tag
        model_folder.mkdir(parents=True, exist_ok=True)

        with (model_folder / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(self.get_metadata(), file, indent=2, sort_keys=True)
        torch.save(self.state_dict(), model_folder / "weights.pt")

    @classmethod
    def load(cls, tag: str = DEFAULT_FACE_CROPPER_TAG) -> FaceCropper:
        """Load a saved cropper model."""
        model_folder = get_models_folder() / tag
        metadata_path = model_folder / "metadata.json"
        weights_path = model_folder / "weights.pt"
        if not metadata_path.is_file() or not weights_path.is_file():
            msg = f"Face cropper artifacts for '{tag}' were not found in {model_folder}."
            raise FileNotFoundError(msg)

        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)

        model = cls(**metadata, pretrained=False)
        model.eval()
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        return model


def load_face_boxes(path: Path) -> dict[str, FaceBox]:
    """Load face box annotations from JSON."""
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        msg = f"Expected a JSON object in {path}."
        raise TypeError(msg)

    face_boxes = {}
    for image_tag, box_payload in payload.items():
        if not isinstance(image_tag, str):
            msg = f"Face box tag must be a string, got {type(image_tag)}."
            raise TypeError(msg)
        if not isinstance(box_payload, dict):
            msg = f"Face box for '{image_tag}' must be an object."
            raise TypeError(msg)
        face_boxes[image_tag] = FaceBox.from_mapping(box_payload)
    return face_boxes
