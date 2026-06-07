"""Endpoints for data operations."""

from functools import lru_cache
from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from minifigures_model.constants import get_data_folder
from minifigures_model.data_utils import FIXED_LR_PREDICTIONS_PATH, load_json

router = APIRouter()


@router.get(
    "/get_image/",
    tags=["data"],
    response_model=BaseModel,
    response_class=FileResponse,
    responses={
        200: {"content": {"image/png": {}}},
        404: {"content": {"application/json": {}}, "description": "Image Not Found"},
    },
)
def get_image(tag: str) -> FileResponse:
    """Get an image from the database."""
    path = get_data_folder() / f"minifigures/{tag}.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image Not Found")
    return FileResponse(path)


@router.get(
    "/get_image_tags/",
    tags=["data"],
    response_model=list[str],
    response_class=JSONResponse,
    responses={200: {"model": list[str]}},
)
def get_image_tags() -> list[str]:
    """Get all image tags in index file."""
    files = (get_data_folder() / "minifigures").glob("*.png")
    return sorted([file.stem for file in files if not file.name.startswith(".")])


@router.get(
    "/get_prediction/",
    tags=["data"],
    response_model=dict[str, float],
    response_class=JSONResponse,
    responses={
        200: {"model": dict[str, float]},
        404: {"content": {"application/json": {}}, "description": "Prediction Not Found"},
    },
)
def get_prediction(tag: str) -> dict[str, float]:
    """Get precomputed model predictions for an image tag."""
    predictions = _load_fixed_lr_predictions()
    prediction = predictions.get(tag)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction Not Found")
    return prediction


@lru_cache
def _load_fixed_lr_predictions() -> dict[str, dict[str, float]]:
    """Load and validate the fixed LR predictions artifact."""
    if not FIXED_LR_PREDICTIONS_PATH.is_file():
        raise HTTPException(status_code=404, detail="Predictions file not found")

    payload = load_json(FIXED_LR_PREDICTIONS_PATH)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Predictions file has an invalid format")

    raw_predictions = payload.get("predictions")
    if not isinstance(raw_predictions, dict):
        raise HTTPException(status_code=500, detail="Predictions file is missing predictions")

    return _validate_predictions(raw_predictions)


def _validate_predictions(raw_predictions: dict[Any, Any]) -> dict[str, dict[str, float]]:
    predictions: dict[str, dict[str, float]] = {}
    for tag, raw_prediction in raw_predictions.items():
        if not isinstance(tag, str) or not isinstance(raw_prediction, dict):
            raise HTTPException(status_code=500, detail="Predictions file contains invalid entries")
        predictions[tag] = _validate_prediction_scores(raw_prediction)
    return predictions


def _validate_prediction_scores(raw_prediction: dict[Any, Any]) -> dict[str, float]:
    prediction: dict[str, float] = {}
    for label, value in raw_prediction.items():
        if not isinstance(label, str) or not isinstance(value, (int, float)):
            raise HTTPException(status_code=500, detail="Predictions file contains invalid scores")
        prediction[label] = float(value)
    return prediction
