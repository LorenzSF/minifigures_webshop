"""Populate Label Studio prediction scores for loss-driven active learning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch import nn
from tqdm import tqdm

try:
    from label_studio_utils import add_prediction_scores_to_label_studio
except ModuleNotFoundError:
    from src.labeling.label_studio_utils import add_prediction_scores_to_label_studio

from minifigures_model.data_utils import (
    IMAGES_DIR,
    MERGED_DATASET_PATH,
    get_latest_dataset_path,
    list_image_paths,
    load_dataset,
)
from minifigures_model.model import EncoderDecoder
from minifigures_model.preprocessing import preprocess_pil_image

DEFAULT_MODEL_TAG = "my_model_active_learning"
DEFAULT_BUDGET = 50


def load_model(model_tag: str) -> EncoderDecoder:
    """Load a saved model in eval mode."""
    model = EncoderDecoder.load(model_tag)
    model.eval()
    return model


def get_embeddings(model_tag: str = DEFAULT_MODEL_TAG) -> dict[str, list[float]]:
    """Get embeddings of all images using the encoder of a saved model."""
    model = load_model(model_tag)
    embeddings = {}

    with torch.inference_mode():
        # Encode every available image.
        for image_path in list_image_paths():
            with Image.open(image_path) as image:
                image_t = preprocess_pil_image(image, resolution=model.resolution)
            embedding = model.encoder(image_t[None,])[0].detach().cpu().tolist()
            embeddings[image_path.stem] = embedding

    return embeddings


def get_loss_trainset(
    train_dataset_path: Path | None = None, model_tag: str = DEFAULT_MODEL_TAG
) -> dict[str, float]:
    """Get the per-image train loss for a saved model."""
    train_dataset_path = train_dataset_path or get_latest_dataset_path("train_seed_*.json")
    dataset = load_dataset(train_dataset_path)
    model = load_model(model_tag)
    cls_to_idx = {class_name: idx for idx, class_name in enumerate(model.classes)}
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    losses = {}

    with torch.inference_mode():
        # Score each labeled training image.
        for tag, labels in dataset.items():
            with Image.open(IMAGES_DIR / f"{tag}.png") as image:
                image_t = preprocess_pil_image(image, resolution=model.resolution)

            target = torch.zeros(len(model.classes), dtype=torch.float32)
            for label in labels:
                if label not in cls_to_idx:
                    raise ValueError(f"Unknown label '{label}' for model '{model.tag}'.")
                target[cls_to_idx[label]] = 1.0

            logits = model(image_t[None,])[0]
            losses[tag] = float(loss_fn(logits, target).mean().item())

    return losses


def get_tags_labeled(merged_dataset_path: Path = MERGED_DATASET_PATH) -> list[str]:
    """Get all tags of images that are already labeled."""
    return sorted(load_dataset(merged_dataset_path))


def add_active_learning_scores(
    model_version: str,
    token: str,
    project_id: int,
    model_tag: str = DEFAULT_MODEL_TAG,
    train_dataset_path: Path | None = None,
    merged_dataset_path: Path = MERGED_DATASET_PATH,
    k: int = 5,
    budget: int = DEFAULT_BUDGET,
) -> None:
    """Add train-loss-driven scores to nearby unlabeled images in Label Studio."""
    labeled_tags = set(get_tags_labeled(merged_dataset_path=merged_dataset_path))
    train_dataset_path = train_dataset_path or get_latest_dataset_path("train_seed_*.json")
    print(f"train_dataset_path: {train_dataset_path}")

    print("Computing loss on the trainset...\n")
    losses_by_tag = get_loss_trainset(train_dataset_path=train_dataset_path, model_tag=model_tag)
    sorted_losses = sorted(losses_by_tag.items(), key=lambda item: item[1], reverse=True)
    train_tags = [tag for tag, _ in sorted_losses]
    train_losses = [loss for _, loss in sorted_losses]

    print("Initializing k nearest neighbors...\n")
    embeddings_index = get_embeddings(model_tag=model_tag)
    unlabeled_tags = [tag for tag in embeddings_index if tag not in labeled_tags]
    if not unlabeled_tags:
        raise ValueError("No unlabeled images remain for active learning.")

    unlabeled_embeddings = np.array([embeddings_index[tag] for tag in unlabeled_tags])
    train_embeddings = np.array([embeddings_index[tag] for tag in train_tags])

    nn_model = NearestNeighbors(n_neighbors=min(k, len(unlabeled_tags)), metric="cosine")
    nn_model.fit(unlabeled_embeddings)

    propagated_scores: dict[str, float] = {}

    # Propagate high training losses to nearby unlabeled images.
    for embedding, loss in tqdm(
        zip(train_embeddings, train_losses, strict=True),
        desc="Computing nearest neighbor for every train image...",
    ):
        knn = nn_model.kneighbors([embedding], return_distance=False)[0, :]
        for neighbor_idx in knn:
            neighbor_tag = unlabeled_tags[neighbor_idx]
            propagated_scores[neighbor_tag] = max(propagated_scores.get(neighbor_tag, 0.0), loss)

    top_scores = dict(
        sorted(propagated_scores.items(), key=lambda item: item[1], reverse=True)[:budget]
    )
    add_prediction_scores_to_label_studio(
        top_scores, model_version=model_version, token=token, project_id=project_id
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--project-id", required=True, type=int)
    parser.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--train-dataset-path", type=Path)
    parser.add_argument("--merged-dataset-path", type=Path, default=MERGED_DATASET_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the active learning command."""
    args = parse_args()
    add_active_learning_scores(
        model_version=args.model_version,
        token=args.token,
        project_id=args.project_id,
        model_tag=args.model_tag,
        train_dataset_path=args.train_dataset_path,
        merged_dataset_path=args.merged_dataset_path,
        k=args.k,
        budget=args.budget,
    )


if __name__ == "__main__":
    main()
