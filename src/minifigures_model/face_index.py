"""Catalog face embedding index utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from minifigures_model.constants import get_data_folder
from minifigures_model.face_cropper import FaceBox
from minifigures_model.face_similarity import (
    DEFAULT_FACE_EMBEDDING_MODEL_TAG,
    SimilarFace,
    rank_similar_faces,
)

if TYPE_CHECKING:
    from pathlib import Path

FACE_INDEX_DIR = get_data_folder() / "face_index"
FACE_EMBEDDINGS_PATH = FACE_INDEX_DIR / "face_embeddings.pt"


@dataclass(frozen=True)
class IndexedFace:
    """Catalog face entry."""

    tag: str
    box: FaceBox
    embedding: torch.Tensor


@dataclass(frozen=True)
class FaceIndex:
    """Loaded catalog face embedding index."""

    entries: dict[str, IndexedFace]
    embedding_model_tag: str = DEFAULT_FACE_EMBEDDING_MODEL_TAG
    cropper_tag: str = "face_cropper"

    @property
    def embeddings(self) -> dict[str, torch.Tensor]:
        """Return tag to embedding mapping."""
        return {tag: entry.embedding for tag, entry in self.entries.items()}

    def find_matches(self, query_embedding: torch.Tensor, *, top_k: int) -> list[SimilarFace]:
        """Return the nearest catalog faces for a query embedding."""
        return rank_similar_faces(query_embedding, self.embeddings, top_k=top_k)

    @classmethod
    def load(cls, path: Path = FACE_EMBEDDINGS_PATH) -> FaceIndex:
        """Load and validate a face index artifact."""
        if not path.is_file():
            msg = f"Face index artifact was not found: {path}"
            raise FileNotFoundError(msg)

        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            msg = f"Expected a dictionary face index payload in {path}."
            raise TypeError(msg)
        return cls.from_payload(payload)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FaceIndex:
        """Create an index from a deserialized payload."""
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, dict):
            msg = "Face index payload is missing an 'entries' object."
            raise TypeError(msg)

        entries = {}
        for tag, raw_entry in raw_entries.items():
            if not isinstance(tag, str):
                msg = f"Face index tag must be a string, got {type(tag)}."
                raise TypeError(msg)
            if not isinstance(raw_entry, dict):
                msg = f"Face index entry for '{tag}' must be an object."
                raise TypeError(msg)
            entries[tag] = _indexed_face_from_payload(tag, raw_entry)

        if not entries:
            msg = "Face index payload does not contain any entries."
            raise ValueError(msg)

        embedding_model_tag = payload.get("embedding_model_tag", DEFAULT_FACE_EMBEDDING_MODEL_TAG)
        cropper_tag = payload.get("cropper_tag", "face_cropper")
        if not isinstance(embedding_model_tag, str) or not isinstance(cropper_tag, str):
            msg = "Face index model tags must be strings."
            raise TypeError(msg)

        return cls(
            entries=entries, embedding_model_tag=embedding_model_tag, cropper_tag=cropper_tag
        )


def _indexed_face_from_payload(tag: str, payload: dict[str, Any]) -> IndexedFace:
    raw_box = payload.get("box")
    raw_embedding = payload.get("embedding")
    if not isinstance(raw_box, dict):
        msg = f"Face index entry for '{tag}' is missing a box object."
        raise TypeError(msg)
    if raw_embedding is None:
        msg = f"Face index entry for '{tag}' is missing an embedding."
        raise TypeError(msg)

    embedding = torch.as_tensor(raw_embedding, dtype=torch.float32)
    if embedding.ndim != 1:
        msg = f"Embedding for '{tag}' must be one-dimensional."
        raise ValueError(msg)

    return IndexedFace(tag=tag, box=FaceBox.from_mapping(raw_box), embedding=embedding)


def save_face_index(
    entries: dict[str, IndexedFace],
    path: Path = FACE_EMBEDDINGS_PATH,
    *,
    embedding_model_tag: str = DEFAULT_FACE_EMBEDDING_MODEL_TAG,
    cropper_tag: str = "face_cropper",
) -> Path:
    """Save a face index artifact."""
    if not entries:
        msg = "Cannot save an empty face index."
        raise ValueError(msg)

    payload = {
        "embedding_model_tag": embedding_model_tag,
        "cropper_tag": cropper_tag,
        "entries": {
            tag: {"box": entry.box.to_dict(), "embedding": entry.embedding.detach().cpu().tolist()}
            for tag, entry in entries.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path
