"""Face embedding and similarity helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as torch_functional

from minifigures_model.preprocessing import preprocess_pil_image

if TYPE_CHECKING:
    from PIL import Image

    from minifigures_model.model import EncoderDecoder

DEFAULT_FACE_EMBEDDING_MODEL_TAG = "my_model"


@dataclass(frozen=True)
class SimilarFace:
    """Ranked face similarity result."""

    tag: str
    score: float


def embed_face(image: Image.Image, model: EncoderDecoder) -> torch.Tensor:
    """Return a normalized encoder embedding for a cropped face image."""
    model.eval()
    with torch.inference_mode():
        image_t = preprocess_pil_image(image, resolution=model.resolution)
        embedding = model.encoder(image_t[None,])[0].detach().cpu()
    return normalize_embedding(embedding)


def normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Normalize a one-dimensional embedding for cosine similarity."""
    if embedding.ndim != 1:
        msg = f"Expected a one-dimensional embedding, got shape {tuple(embedding.shape)}."
        raise ValueError(msg)
    return torch_functional.normalize(embedding.to(dtype=torch.float32), dim=0)


def cosine_similarity(query_embedding: torch.Tensor, candidate_embedding: torch.Tensor) -> float:
    """Compute cosine similarity between two one-dimensional embeddings."""
    query = normalize_embedding(query_embedding)
    candidate = normalize_embedding(candidate_embedding)
    if query.shape != candidate.shape:
        msg = f"Embedding shapes do not match: {tuple(query.shape)} != {tuple(candidate.shape)}."
        raise ValueError(msg)
    return float(torch.dot(query, candidate).item())


def rank_similar_faces(
    query_embedding: torch.Tensor, candidate_embeddings: dict[str, torch.Tensor], *, top_k: int
) -> list[SimilarFace]:
    """Rank candidate face embeddings by cosine similarity."""
    if top_k <= 0:
        msg = f"top_k must be positive, got {top_k}."
        raise ValueError(msg)
    if not candidate_embeddings:
        msg = "Face index does not contain any embeddings."
        raise ValueError(msg)

    matches = [
        SimilarFace(tag=tag, score=cosine_similarity(query_embedding, candidate_embedding))
        for tag, candidate_embedding in candidate_embeddings.items()
    ]
    return sorted(matches, key=lambda match: match.score, reverse=True)[:top_k]
