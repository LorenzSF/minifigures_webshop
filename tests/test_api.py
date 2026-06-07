"""Minifigures Webshop REST API test suite."""

import io
from http import HTTPStatus
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from starlette.exceptions import HTTPException

import minifigures_api.routers.data as data_router_module
import minifigures_api.routers.face_search as face_search_router_module
import minifigures_api.routers.predict as predict_router_module
from minifigures_api.api import app
from minifigures_model.face_cropper import FaceBox
from minifigures_model.face_similarity import SimilarFace

client = TestClient(app)


def test_read_root() -> None:
    """Test that reading the root is successful."""
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK


def test_get_image_tags_returns_list() -> None:
    """Test that image tags endpoint returns a JSON list."""
    response = client.get("/data/get_image_tags/")
    assert response.status_code == HTTPStatus.OK
    assert isinstance(response.json(), list)


def test_get_image_missing_tag_returns_404() -> None:
    """Test that requesting a missing image returns 404."""
    response = client.get("/data/get_image/", params={"tag": "__missing_tag__"})
    assert response.status_code == HTTPStatus.NOT_FOUND


def test_get_prediction_returns_precomputed_scores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that precomputed catalog predictions are served by image tag."""
    predictions_path = tmp_path / "all_predictions_fixed_lr.json"
    predictions_path.write_text(
        '{"predictions": {"sw0001": {"helmet": 0.8, "robot": 0.2}}}', encoding="utf-8"
    )
    monkeypatch.setattr(data_router_module, "FIXED_LR_PREDICTIONS_PATH", predictions_path)
    data_router_module._load_fixed_lr_predictions.cache_clear()

    response = client.get("/data/get_prediction/", params={"tag": "sw0001"})

    data_router_module._load_fixed_lr_predictions.cache_clear()
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"helmet": 0.8, "robot": 0.2}


def test_get_prediction_missing_tag_returns_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that unknown prediction tags return 404."""
    predictions_path = tmp_path / "all_predictions_fixed_lr.json"
    predictions_path.write_text('{"predictions": {}}', encoding="utf-8")
    monkeypatch.setattr(data_router_module, "FIXED_LR_PREDICTIONS_PATH", predictions_path)
    data_router_module._load_fixed_lr_predictions.cache_clear()

    response = client.get("/data/get_prediction/", params={"tag": "__missing_tag__"})

    data_router_module._load_fixed_lr_predictions.cache_clear()
    assert response.status_code == HTTPStatus.NOT_FOUND


def test_predict_rejects_invalid_image_bytes() -> None:
    """Test that corrupt image uploads return an HTTP 415."""
    response = client.post(
        "/predict/image/", files={"file": ("sample.png", b"not a real image", "image/png")}
    )
    assert response.status_code == HTTPStatus.UNSUPPORTED_MEDIA_TYPE


def test_predict_returns_404_when_model_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that predict endpoint surfaces a missing model as HTTP 404."""

    def _raise_missing_model() -> None:
        raise HTTPException(status_code=404, detail="Model 'my_model' not found")

    # Patch the function imported in the router module.
    monkeypatch.setattr(predict_router_module, "fetch_model", _raise_missing_model)

    image = Image.new("RGB", (1, 1), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()

    response = client.post("/predict/image/", files={"file": ("sample.png", payload, "image/png")})
    assert response.status_code == HTTPStatus.NOT_FOUND


def test_face_search_rejects_invalid_image_bytes() -> None:
    """Test that corrupt face search uploads return an HTTP 415."""
    response = client.post(
        "/face/search/", files={"file": ("sample.png", b"not a real image", "image/png")}
    )
    assert response.status_code == HTTPStatus.UNSUPPORTED_MEDIA_TYPE


def test_face_search_returns_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the face search response contract with mocked model artifacts."""

    class FakeCropper:
        def predict_box(self, _image: Image.Image) -> FaceBox:
            return FaceBox(x_center=0.5, y_center=0.5, width=0.5, height=0.5)

    class FakeIndex:
        embedding_model_tag = "my_model"

        def find_matches(self, _query_embedding: torch.Tensor, *, top_k: int) -> list[SimilarFace]:
            assert top_k == 1
            return [SimilarFace(tag="sw0001", score=0.9)]

    def fetch_fake_cropper() -> FakeCropper:
        return FakeCropper()

    def fetch_fake_index() -> FakeIndex:
        return FakeIndex()

    def fetch_fake_embedding_model(_tag: str) -> object:
        return object()

    def embed_fake_face(_image: Image.Image, _model: object) -> torch.Tensor:
        return torch.tensor([1.0, 0.0])

    monkeypatch.setattr(face_search_router_module, "fetch_face_cropper", fetch_fake_cropper)
    monkeypatch.setattr(face_search_router_module, "fetch_face_index", fetch_fake_index)
    monkeypatch.setattr(
        face_search_router_module, "fetch_face_embedding_model", fetch_fake_embedding_model
    )
    monkeypatch.setattr(face_search_router_module, "embed_face", embed_fake_face)

    image = Image.new("RGB", (20, 20), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    response = client.post(
        "/face/search/",
        params={"top_k": 1},
        files={"file": ("sample.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["matches"] == [{"tag": "sw0001", "score": 0.9}]


def test_face_search_returns_404_when_artifact_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that face search surfaces missing artifacts as HTTP 404."""

    def _raise_missing_cropper() -> None:
        raise HTTPException(status_code=404, detail="Face cropper not found")

    monkeypatch.setattr(face_search_router_module, "fetch_face_cropper", _raise_missing_cropper)

    image = Image.new("RGB", (20, 20), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    response = client.post(
        "/face/search/", files={"file": ("sample.png", buffer.getvalue(), "image/png")}
    )

    assert response.status_code == HTTPStatus.NOT_FOUND
