"""Minifigures Webshop REST API test suite."""

import io

from fastapi.testclient import TestClient
from PIL import Image
from starlette.exceptions import HTTPException

import minifigures_api.routers.predict as predict_router_module
from minifigures_api.api import app

client = TestClient(app)


def test_read_root() -> None:
    """Test that reading the root is successful."""
    response = client.get("/")
    assert response.status_code == 200


def test_get_image_tags_returns_list() -> None:
    """Test that image tags endpoint returns a JSON list."""
    response = client.get("/data/get_image_tags/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_image_missing_tag_returns_404() -> None:
    """Test that requesting a missing image returns 404."""
    response = client.get("/data/get_image/", params={"tag": "__missing_tag__"})
    assert response.status_code == 404


def test_predict_returns_404_when_model_is_missing(monkeypatch) -> None:
    """Test that predict endpoint surfaces a missing model as HTTP 404."""

    def _raise_missing_model() -> None:
        raise HTTPException(status_code=404, detail="Model 'my_model' not found")

    # Patch the function imported in the router module.
    monkeypatch.setattr(predict_router_module, "fetch_model", _raise_missing_model)

    image = Image.new("RGB", (1, 1), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()

    response = client.post(
        "/predict/image/",
        files={"file": ("sample.png", payload, "image/png")},
    )
    assert response.status_code == 404
