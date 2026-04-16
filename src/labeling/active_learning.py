"""Populate Label Studio prediction scores for active learning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from config import HOST, PORT
from label_studio_sdk import Client
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch import nn
from tqdm import tqdm

from minifigures_model.model import EncoderDecoder
from minifigures_model.model_finetune import (
    get_classes,
    get_latest_dataset_path,
    load_dataset,
    preprocess_pil_image,
)

DATA_DIR = Path("/workspaces/updated-minifigures-webshop-2026-LorenzSF/data/data")
IMAGES_DIR = DATA_DIR / "minifigures"
MERGED_DATASET_PATH = DATA_DIR / "dataset_merged_labeled.json"
DEFAULT_MODEL_TAG = "my_model_active_learning"
DEFAULT_BUDGET = 50


def add_prediction_scores_to_label_studio(
    tags_scores: dict[str, float], model_version: str, token: str, project_id: int
) -> None:
    """Add prediction score attribute to Label Studio tasks."""
    ls = Client(url=f"{HOST}:{PORT}", api_key=token)
    project = ls.get_project(project_id)

    assert model_version not in project.get_model_versions(), (
        "The model_version you provided, already exists. Choose a different one."
    )

    print("Setting prediction scores in label-studio...")
    for task in project.get_tasks():
        image_path = Path(task["storage_filename"])
        tag = image_path.stem
        if tag in tags_scores:
            project.create_prediction(
                task["id"], result=[], score=tags_scores[tag], model_version=model_version
            )


def _load_model(model_tag: str) -> EncoderDecoder:
    """Load a saved model in eval mode on CPU."""
    model = EncoderDecoder.load(model_tag)
    model.eval()
    return model


def get_embeddings(model_tag: str = DEFAULT_MODEL_TAG) -> dict[str, list[float]]:
    """Get embeddings of all images using the encoder of a saved model."""
    model = _load_model(model_tag)
    embeddings = {}
    with torch.no_grad():
        for image_path in sorted(IMAGES_DIR.glob("*.png")):
            if image_path.name.startswith("._"):
                continue
            image = Image.open(image_path)
            image_t = preprocess_pil_image(image, resolution=model.resolution)
            embedding = model.encoder(image_t[None,])[0].detach().cpu().numpy().tolist()
            embeddings[image_path.stem] = embedding
    return embeddings


def get_loss_trainset(
    train_dataset_path: Path | None = None, model_tag: str = DEFAULT_MODEL_TAG
) -> dict[str, float]:
    """Get the per-image train loss for a saved model."""
    train_dataset_path = train_dataset_path or get_latest_dataset_path("train_seed_*.json")
    dataset = load_dataset(train_dataset_path)
    classes = get_classes(dataset)
    cls_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    model = _load_model(model_tag)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    losses = {}

    with torch.no_grad():
        for tag, labels in dataset.items():
            image = Image.open(IMAGES_DIR / f"{tag}.png")
            image_t = preprocess_pil_image(image, resolution=model.resolution)
            logits = model(image_t[None,])[0]
            target = torch.zeros(len(classes), dtype=torch.float32)
            for label in labels:
                target[cls_to_idx[label]] = 1.0
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
    tags_all_label = get_tags_labeled(merged_dataset_path=merged_dataset_path)
    train_dataset_path = train_dataset_path or get_latest_dataset_path("train_seed_*.json")
    print(f"train_dataset_path: {train_dataset_path}")

    print("Computing loss on the trainset...\n")
    losses_tags = get_loss_trainset(train_dataset_path=train_dataset_path, model_tag=model_tag)
    losses, tags = list(losses_tags.values()), list(losses_tags.keys())
    sorter = np.argsort(np.array(losses))[::-1]
    tags = [tags[idx] for idx in sorter]
    losses = [losses[idx] for idx in sorter]

    print("Initializing k nearest neighbors...\n")
    index = get_embeddings(model_tag=model_tag)
    embeddings_nolabel = np.array([index[tag] for tag in index if tag not in tags_all_label])
    tags_nolabel = [tag for tag in index if tag not in tags_all_label]
    embeddings_label = np.array([index[tag] for tag in tags])

    if len(tags_nolabel) == 0:
        raise ValueError("No unlabeled images remain for active learning.")

    nn_model = NearestNeighbors(n_neighbors=min(k, len(tags_nolabel)), metric="cosine")
    nn_model.fit(embeddings_nolabel)

    tags_nn_loss: dict[str, float] = {}
    for embedding, loss in tqdm(
        zip(embeddings_label, losses, strict=True),
        desc="Computing nearest neighbor for every train image...",
    ):
        knn = nn_model.kneighbors([embedding], return_distance=False)[0, :]
        for ind_knn in knn:
            tag_nn = tags_nolabel[ind_knn]
            tags_nn_loss[tag_nn] = max(tags_nn_loss.get(tag_nn, 0.0), round(loss, 4))

    tags_nn_loss = dict(
        sorted(tags_nn_loss.items(), key=lambda item: item[1], reverse=True)[:budget]
    )
    add_prediction_scores_to_label_studio(
        tags_nn_loss, model_version=model_version, token=token, project_id=project_id
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


if __name__ == "__main__":
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
