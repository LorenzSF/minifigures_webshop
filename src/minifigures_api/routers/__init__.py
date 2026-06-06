"""API routers."""

from minifigures_api.routers.data import router as data_router
from minifigures_api.routers.face_search import router as face_search_router
from minifigures_api.routers.predict import router as predict_router
from minifigures_api.routers.utils import fetch_model

__all__ = ["data_router", "face_search_router", "fetch_model", "predict_router"]
