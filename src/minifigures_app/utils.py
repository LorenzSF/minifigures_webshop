"""Utils functions for the Streamlit app."""

import io
from typing import TypedDict

import requests
from PIL import Image
from requests import RequestException

from minifigures_app.constants import URL

REQUEST_TIMEOUT_SECONDS = 10


class ApiResponseError(RuntimeError):
    """Raised when the API response cannot be used by the Streamlit app."""


class FaceMatchPayload(TypedDict):
    """Face search match payload returned by the API."""

    tag: str
    score: float


class FaceSearchPayload(TypedDict):
    """Face search payload returned by the API."""

    query_box: dict[str, float]
    matches: list[FaceMatchPayload]


def _image_to_png_bytes(image: Image.Image) -> bytes:
    """Serialize a PIL image as PNG bytes."""
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    return buff.getvalue()


def predict_image(image: Image.Image) -> dict[str, float]:
    """Get model predictions for a given image using a FastAPI request."""
    img_bytes = _image_to_png_bytes(image)

    # Create the prediction
    try:
        response = requests.post(
            url=f"{URL}/predict/image/",
            files=[("file", ("UID", img_bytes, "image/png"))],
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except RequestException as exc:
        msg = f"Prediction request failed: {exc}"
        raise ApiResponseError(msg) from exc
    except ValueError as exc:
        msg = "Prediction response is not valid JSON"
        raise ApiResponseError(msg) from exc

    # Return the result
    if "prediction" not in payload:
        msg = "Prediction response is missing the 'prediction' field"
        raise ApiResponseError(msg)
    return payload["prediction"]


def get_prediction(tag: str) -> dict[str, float]:
    """Get precomputed model predictions for an image tag."""
    try:
        response = requests.get(
            url=f"{URL}/data/get_prediction/", params={"tag": tag}, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        payload = response.json()
    except RequestException as exc:
        msg = f"Failed to fetch prediction for '{tag}': {exc}"
        raise ApiResponseError(msg) from exc
    except ValueError as exc:
        msg = "Prediction response is not valid JSON"
        raise ApiResponseError(msg) from exc

    return _validate_prediction_payload(payload)


def _validate_prediction_payload(payload: object) -> dict[str, float]:
    """Validate a prediction mapping returned by the API."""
    if not isinstance(payload, dict):
        msg = "Prediction response has an invalid format"
        raise ApiResponseError(msg)

    prediction: dict[str, float] = {}
    for label, value in payload.items():
        if not isinstance(label, str) or not isinstance(value, (int, float)):
            msg = "Prediction response contains invalid scores"
            raise ApiResponseError(msg)
        prediction[label] = float(value)
    return prediction


def search_similar_faces(image: Image.Image, top_k: int = 6) -> FaceSearchPayload:
    """Search catalog products with faces similar to the provided image."""
    if top_k <= 0:
        msg = f"top_k must be positive, got {top_k}."
        raise ValueError(msg)

    img_bytes = _image_to_png_bytes(image)
    try:
        response = requests.post(
            url=f"{URL}/face/search/",
            params={"top_k": top_k},
            files=[("file", ("UID", img_bytes, "image/png"))],
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except RequestException as exc:
        msg = f"Face search request failed: {exc}"
        raise ApiResponseError(msg) from exc
    except ValueError as exc:
        msg = "Face search response is not valid JSON"
        raise ApiResponseError(msg) from exc

    return _validate_face_search_payload(payload)


def _validate_face_search_payload(payload: object) -> FaceSearchPayload:
    """Validate the face search API payload shape."""
    if not isinstance(payload, dict):
        msg = "Face search response has an invalid format"
        raise ApiResponseError(msg)

    query_box = payload.get("query_box")
    raw_matches = payload.get("matches")
    if not isinstance(query_box, dict):
        msg = "Face search response is missing the 'query_box' field"
        raise ApiResponseError(msg)
    if not isinstance(raw_matches, list):
        msg = "Face search response is missing the 'matches' field"
        raise ApiResponseError(msg)

    matches = []
    for match in raw_matches:
        if not isinstance(match, dict):
            msg = "Face search response contains an invalid match"
            raise ApiResponseError(msg)
        tag = match.get("tag")
        score = match.get("score")
        if not isinstance(tag, str) or not isinstance(score, (int, float)):
            msg = "Face search response contains an invalid match"
            raise ApiResponseError(msg)
        matches.append({"tag": tag, "score": float(score)})

    try:
        query_box_values = {str(key): float(value) for key, value in query_box.items()}
    except (TypeError, ValueError) as exc:
        msg = "Face search response contains an invalid query box"
        raise ApiResponseError(msg) from exc

    return {"query_box": query_box_values, "matches": matches}


def list_im_tags() -> list[str]:
    """Get all image tags in index file."""
    try:
        response = requests.get(url=f"{URL}/data/get_image_tags/", timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except RequestException as exc:
        msg = f"Failed to fetch image tags: {exc}"
        raise ApiResponseError(msg) from exc
    except ValueError as exc:
        msg = "Image tags response is not valid JSON"
        raise ApiResponseError(msg) from exc

    if not isinstance(payload, list):
        msg = "Image tags response has an invalid format"
        raise ApiResponseError(msg)
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
        msg = f"Failed to fetch image '{tag}': {exc}"
        raise ApiResponseError(msg) from exc
    except OSError as exc:
        msg = f"Image '{tag}' payload is not a valid image"
        raise ApiResponseError(msg) from exc
