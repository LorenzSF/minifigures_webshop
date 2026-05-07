"""Tests for Streamlit app API URL configuration."""

import importlib

from minifigures_app import constants


def test_api_url_defaults_to_localhost(monkeypatch) -> None:
    """The app should target the local API when no Docker host is configured."""
    monkeypatch.delenv("API_HOST", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)

    reloaded = importlib.reload(constants)

    assert reloaded.HOST == "http://localhost"
    assert reloaded.PORT == "8000"
    assert reloaded.URL == "http://localhost:8000"


def test_api_url_respects_docker_env(monkeypatch) -> None:
    """An explicit API host should still work for container-based deployments."""
    monkeypatch.setenv("API_HOST", "http://api")
    monkeypatch.setenv("API_PORT", "9000")

    reloaded = importlib.reload(constants)

    assert reloaded.HOST == "http://api"
    assert reloaded.PORT == "9000"
    assert reloaded.URL == "http://api:9000"
