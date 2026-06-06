"""Tests for face similarity helpers."""

import pytest
import torch

from minifigures_model.face_similarity import cosine_similarity, rank_similar_faces


def test_cosine_similarity_rejects_mismatched_shapes() -> None:
    """Cosine similarity should reject incompatible embedding shapes."""
    with pytest.raises(ValueError, match="Embedding shapes"):
        cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0, 0.0]))


def test_rank_similar_faces_orders_by_score() -> None:
    """Ranking should place the nearest candidate first."""
    matches = rank_similar_faces(
        torch.tensor([1.0, 0.0]),
        {"near": torch.tensor([0.9, 0.1]), "far": torch.tensor([0.0, 1.0])},
        top_k=2,
    )

    assert [match.tag for match in matches] == ["near", "far"]
    assert matches[0].score > matches[1].score
