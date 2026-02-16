"""Endpoint for model predictions."""

from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse

from minifigures_api.routers.models import Prediction
from minifigures_api.routers.utils import extract_images_from_files

router = APIRouter()


@router.post(
    "/image/",
    tags=["prediction"],
    response_model=Prediction,
    response_class=JSONResponse,
    responses={200: {"model": Prediction}},
)
def predict(file: UploadFile) -> Prediction:
    """Use the tagged model to predict over an image."""
    # Convert the UploadFile to a torch Tensor
    _file = extract_images_from_files([file])[0]

    # TODO: make a prediction

    # Return the result
    return Prediction(prediction={"class1": 0.42, "class2": 0.42})
