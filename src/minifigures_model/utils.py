"""Compatibility wrappers for older imports."""

import torch
from PIL import Image

from minifigures_model.preprocessing import pil_to_tensor, resize_tensor


def pil_to_torch(x: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a torch tensor."""
    return pil_to_tensor(x)


def resize(x: torch.Tensor, resolution: int) -> torch.Tensor:
    """Resize a single image tensor to a given resolution."""
    return resize_tensor(x, resolution=resolution)
