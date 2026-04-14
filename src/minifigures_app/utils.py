"""Utils functions for the Streamlit app."""

import io

import requests
from PIL import Image
from requests import RequestException

from minifigures_app.constants import URL

REQUEST_TIMEOUT_SECONDS = 10


def predict_image(image: Image.Image) -> dict[str, float]:
    """Get model predictions for a given image using a FastAPI request."""
    # Format the uploaded image
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    img_str = buff.getvalue()

    # Create the prediction
    try:
        response = requests.post(
            url=f"{URL}/predict/image/",
            files=[("file", ("UID", img_str, "image/png"))],
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except RequestException as exc:
        raise RuntimeError(f"Prediction request failed: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("Prediction response is not valid JSON") from exc

    # Return the result
    if "prediction" not in payload:
        raise RuntimeError("Prediction response is missing the 'prediction' field")
    return payload["prediction"]


def list_im_tags() -> list[str]:
    """Get all image tags in index file."""
    try:
        response = requests.get(url=f"{URL}/data/get_image_tags/", timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except RequestException as exc:
        raise RuntimeError(f"Failed to fetch image tags: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("Image tags response is not valid JSON") from exc

    if not isinstance(payload, list):
        raise RuntimeError("Image tags response has an invalid format")
    return payload


def get_image(tag: str) -> Image.Image:
    """Get an image from the database."""
    try:
        response = requests.get(
            url=f"{URL}/data/get_image/", params={"tag": tag}, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        with Image.open(io.BytesIO(response.content)) as image:
            return image.convert("RGB")
    except RequestException as exc:
        raise RuntimeError(f"Failed to fetch image '{tag}': {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Image '{tag}' payload is not a valid image") from exc
