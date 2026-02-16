"""Minifigures Webshop REST API test suite."""

from fastapi.testclient import TestClient

from minifigures_api.api import app

client = TestClient(app)


def test_read_root() -> None:
    """Test that reading the root is successful."""
    response = client.get("/")
    assert response.status_code == 200
