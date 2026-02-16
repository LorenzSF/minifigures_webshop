"""Utilisation functions."""

from fastapi import UploadFile
from PIL import Image
from starlette.exceptions import HTTPException


def extract_images_from_files(files: list[UploadFile]) -> list[Image.Image]:
    """Extract images from a list of files."""
    extracted = []
    for file in files:
        if file.content_type not in ("image/jpeg", "image/png"):
            raise HTTPException(
                status_code=415, detail=f"Media type '{file.content_type}' not valid"
            )
        with Image.open(file.file) as img:
            extracted.append(img.convert("RGB").copy())
    return extracted
