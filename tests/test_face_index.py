"""Tests for face index validation and ranking."""

from pathlib import Path

import torch

from minifigures_model.face_cropper import FaceBox
from minifigures_model.face_index import FaceIndex, IndexedFace, save_face_index


def test_face_index_round_trip(tmp_path: Path) -> None:
    """A saved face index should load back with usable entries."""
    path = tmp_path / "face_embeddings.pt"
    entries = {
        "a": IndexedFace(
            tag="a",
            box=FaceBox(x_center=0.5, y_center=0.2, width=0.2, height=0.2),
            embedding=torch.tensor([1.0, 0.0]),
        )
    }

    save_face_index(entries, path, embedding_model_tag="my_model", cropper_tag="face_cropper")
    face_index = FaceIndex.load(path)

    assert face_index.embedding_model_tag == "my_model"
    assert face_index.cropper_tag == "face_cropper"
    assert list(face_index.entries) == ["a"]


def test_face_index_finds_matches() -> None:
    """Face index should delegate to similarity ranking."""
    face_index = FaceIndex(
        entries={
            "near": IndexedFace(
                tag="near",
                box=FaceBox(x_center=0.5, y_center=0.2, width=0.2, height=0.2),
                embedding=torch.tensor([1.0, 0.0]),
            ),
            "far": IndexedFace(
                tag="far",
                box=FaceBox(x_center=0.5, y_center=0.2, width=0.2, height=0.2),
                embedding=torch.tensor([0.0, 1.0]),
            ),
        }
    )

    matches = face_index.find_matches(torch.tensor([1.0, 0.0]), top_k=1)

    assert len(matches) == 1
    assert matches[0].tag == "near"
