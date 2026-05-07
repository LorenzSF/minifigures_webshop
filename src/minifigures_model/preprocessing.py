"""Shared image preprocessing helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_tensor(x: torch.Tensor | np.ndarray | Image.Image) -> torch.Tensor:
    """Convert supported inputs into a float tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32)
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, dtype=torch.float32)
    if isinstance(x, Image.Image):
        return torch.as_tensor(np.array(x), dtype=torch.float32)
    msg = f"Variable of type '{type(x)}' not supported!"
    raise TypeError(msg)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a channel-first tensor."""
    return to_tensor(image.convert("RGB")).permute(2, 0, 1) / 255.0


def pad_to_square(image: torch.Tensor) -> torch.Tensor:
    """Pad a channel-first image tensor to a square."""
    _, h, w = image.shape
    max_wh = max(w, h)
    p_left, p_right = ((max_wh - w) // 2 for _ in range(2))
    p_right += int(w % 2 == 1)
    p_bottom, p_top = ((max_wh - h) // 2 for _ in range(2))
    p_top += int(h % 2 == 1)
    padding = (p_left, p_right, p_top, p_bottom)
    return F.pad(image, padding, value=1, mode="constant")


def resize_tensor(image: torch.Tensor, resolution: int) -> torch.Tensor:
    """Resize a channel-first tensor to the target resolution."""
    return F.interpolate(image[None,], size=(resolution, resolution), mode="bilinear")[0]


def normalize_tensor(image: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization."""
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (image - mean) / std


def preprocess_image_tensor(image: torch.Tensor, resolution: int = 256) -> torch.Tensor:
    """Apply the shared resize and normalization pipeline."""
    image = pad_to_square(image)
    image = resize_tensor(image, resolution=resolution)
    return normalize_tensor(image)


def preprocess_pil_image(image: Image.Image, resolution: int = 256) -> torch.Tensor:
    """Apply the shared preprocessing pipeline to a PIL image."""
    return preprocess_image_tensor(pil_to_tensor(image), resolution=resolution)
