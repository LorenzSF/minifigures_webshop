"""Endpoint for face similarity search."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Query, UploadFile
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from minifigures_api.routers.basemodels import FaceBox as FaceBoxResponse
from minifigures_api.routers.basemodels import FaceMatch, FaceSearchResponse
from minifigures_api.routers.utils import extract_images_from_files
from minifigures_model.face_cropper import DEFAULT_FACE_CROPPER_TAG, FaceCropper, crop_face
from minifigures_model.face_index import FACE_EMBEDDINGS_PATH, FaceIndex
from minifigures_model.face_similarity import embed_face
from minifigures_model.model import EncoderDecoder

DEFAULT_FACE_SEARCH_TOP_K = 6
MAX_FACE_SEARCH_TOP_K = 24
FACE_SEARCH_INPUT_ERROR_STATUS = 422

router = APIRouter()


@lru_cache
def fetch_face_cropper(tag: str = DEFAULT_FACE_CROPPER_TAG) -> FaceCropper:
    """Fetch a saved face cropper model."""
    try:
        return FaceCropper.load(tag)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@lru_cache
def fetch_face_index(path: str = str(FACE_EMBEDDINGS_PATH)) -> FaceIndex:
    """Fetch the exported face embedding index."""
    try:
        return FaceIndex.load(Path(path))
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@lru_cache
def fetch_face_embedding_model(tag: str) -> EncoderDecoder:
    """Fetch the encoder model used by the face index."""
    try:
        return EncoderDecoder.load(tag)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/search/",
    tags=["face-search"],
    response_model=FaceSearchResponse,
    response_class=JSONResponse,
    responses={
        200: {"model": FaceSearchResponse},
        404: {"content": {"application/json": {}}, "description": "Face search artifact not found"},
        415: {"content": {"application/json": {}}, "description": "Media type not valid"},
        422: {"content": {"application/json": {}}, "description": "Face search input not usable"},
    },
)
def search_faces(
    file: UploadFile,
    top_k: Annotated[int, Query(ge=1, le=MAX_FACE_SEARCH_TOP_K)] = DEFAULT_FACE_SEARCH_TOP_K,
) -> FaceSearchResponse:
    """Search catalog minifigures with faces similar to an uploaded image."""
    image = extract_images_from_files([file])[0]
    cropper = fetch_face_cropper()
    face_index = fetch_face_index()
    embedding_model = fetch_face_embedding_model(face_index.embedding_model_tag)

    try:
        query_box = cropper.predict_box(image)
        query_face = crop_face(image, query_box)
        query_embedding = embed_face(query_face, embedding_model)
        matches = face_index.find_matches(query_embedding, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=FACE_SEARCH_INPUT_ERROR_STATUS, detail=str(exc)) from exc

    return FaceSearchResponse(
        query_box=FaceBoxResponse(**query_box.to_dict()),
        matches=[FaceMatch(tag=match.tag, score=match.score) for match in matches],
    )
