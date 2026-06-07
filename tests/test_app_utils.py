"""Tests for Streamlit app utility validation."""

import pytest
from PIL import Image

from minifigures_app import utils


class FakeResponse:
    """Minimal requests response test double."""

    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        """Simulate a successful response."""

    def json(self) -> object:
        """Return the configured payload."""
        return self.payload


def test_search_similar_faces_validates_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Face search client should return validated matches."""

    def fake_post(**kwargs: object) -> FakeResponse:
        assert kwargs["params"] == {"top_k": 2}
        return FakeResponse(
            {
                "query_box": {"x_center": 0.5, "y_center": 0.2, "width": 0.2, "height": 0.2},
                "matches": [{"tag": "sw0001", "score": 0.9}],
            }
        )

    monkeypatch.setattr(utils.requests, "post", fake_post)

    payload = utils.search_similar_faces(Image.new("RGB", (4, 4)), top_k=2)

    assert payload["matches"] == [{"tag": "sw0001", "score": 0.9}]


def test_get_prediction_validates_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Precomputed prediction client should return validated scores."""

    def fake_get(**kwargs: object) -> FakeResponse:
        assert kwargs["params"] == {"tag": "sw0001"}
        return FakeResponse({"helmet": 0.8, "robot": 1})

    monkeypatch.setattr(utils.requests, "get", fake_get)

    assert utils.get_prediction("sw0001") == {"helmet": 0.8, "robot": 1.0}


def test_search_similar_faces_rejects_missing_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Face search client should reject malformed API payloads."""

    def fake_post(**_kwargs: object) -> FakeResponse:
        return FakeResponse({"query_box": {}})

    monkeypatch.setattr(utils.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="matches"):
        utils.search_similar_faces(Image.new("RGB", (4, 4)))
