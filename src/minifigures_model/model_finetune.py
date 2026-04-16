"""Dataset preparation and finetuning utilities based on the course notebook."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skmultilearn.model_selection import iterative_train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from minifigures_model.model import EncoderDecoder

DATA_DIR = Path("/workspaces/updated-minifigures-webshop-2026-LorenzSF/data/data")
IMAGES_DIR = DATA_DIR / "minifigures"
MERGED_DATASET_PATH = DATA_DIR / "dataset_merged_labeled.json"
DATASETS_DIR = DATA_DIR / "datasets"
DEFAULT_BASE_MODEL_TAG = "my_model"
DEFAULT_OUTPUT_MODEL_TAG = "my_model_active_learning"


def load_dataset(path: Path) -> dict[str, list[str]]:
    """Load a JSON dataset file."""
    with open(path) as f:
        return json.load(f)


def save_dataset(dataset: dict[str, list[str]], path: Path) -> None:
    """Save a JSON dataset file."""
    with open(path, "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)


def get_latest_dataset_path(pattern: str, data_dir: Path = DATASETS_DIR) -> Path:
    """Return the newest dataset split matching the provided glob pattern."""
    candidates = sorted(data_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        msg = f"No dataset files found for pattern: {pattern}"
        raise FileNotFoundError(msg)
    return candidates[0]


def get_classes(dataset: dict[str, list[str]]) -> list[str]:
    """Infer the sorted class list from a dataset mapping."""
    return sorted({label for labels in dataset.values() for label in labels})


def to_tensor(x: torch.Tensor | np.ndarray | Image.Image) -> torch.Tensor:
    """Convert supported inputs into a tensor."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.Tensor(x)
    if isinstance(x, Image.Image):
        return torch.Tensor(np.array(x))
    msg = f"Variable of type '{type(x)}' not supported!"
    raise TypeError(msg)


def load_img(path: Path) -> torch.Tensor:
    """Load an image into a channel-first float tensor."""
    image = Image.open(path).convert("RGB")
    image = to_tensor(image)
    return image.permute(2, 0, 1) / 255.0


class SquarePad:
    """Pad a channel-first image tensor to a square."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Pad the provided image to become a square."""
        _, _, h, w = x.shape
        max_wh = max(w, h)
        p_left, p_right = ((max_wh - w) // 2 for _ in range(2))
        p_right += int(w % 2 == 1)
        p_bottom, p_top = ((max_wh - h) // 2 for _ in range(2))
        p_top += int(h % 2 == 1)
        padding = (p_left, p_right, p_top, p_bottom)
        return F.pad(x, padding, value=1, mode="constant")


def get_transform_resize(resolution: int) -> transforms.Compose:
    """Generate the resizing transform from the notebook."""
    return transforms.Compose([SquarePad(), transforms.Resize((resolution, resolution))])


def get_transform_normalize() -> transforms.Compose:
    """Generate the normalization transform from the notebook."""
    return transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def preprocess_image_tensor(image: torch.Tensor, resolution: int = 256) -> torch.Tensor:
    """Apply notebook-style resize and normalization to an image tensor."""
    image = get_transform_resize(resolution)(image[None,])[0]
    return get_transform_normalize()(image)


def preprocess_pil_image(image: Image.Image, resolution: int = 256) -> torch.Tensor:
    """Apply notebook-style preprocessing to a PIL image."""
    image_t = to_tensor(image.convert("RGB")).permute(2, 0, 1) / 255.0
    return preprocess_image_tensor(image_t, resolution=resolution)


def _binarize_labels(
    tags: list[str], dataset: dict[str, list[str]], classes: list[str]
) -> np.ndarray:
    """Convert multilabel strings into a binary matrix."""
    matrix = np.zeros((len(tags), len(classes)), dtype=int)
    cls_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    for row_idx, tag in enumerate(tags):
        for label in dataset[tag]:
            matrix[row_idx, cls_to_idx[label]] = 1
    return matrix


def _subset_from_tags(dataset: dict[str, list[str]], tags: list[str]) -> dict[str, list[str]]:
    """Take a subset of a dataset preserving labels."""
    return {tag: dataset[tag] for tag in tags}


def create_seed_splits(
    source_dataset_path: Path = MERGED_DATASET_PATH,
    output_dir: Path = DATASETS_DIR,
    seed_size: int = 50,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> dict[str, Path]:
    """Create an initial seed dataset and its train/val split."""
    dataset = load_dataset(source_dataset_path)
    classes = get_classes(dataset)
    tags = sorted(dataset)

    if seed_size <= 0 or seed_size >= len(tags):
        msg = f"seed_size must be between 1 and {len(tags) - 1}, got {seed_size}"
        raise ValueError(msg)

    x_all = np.array(tags, dtype=object).reshape(-1, 1)
    y_all = _binarize_labels(tags, dataset, classes)
    _, _, x_seed, _ = iterative_train_test_split(x_all, y_all, test_size=seed_size / len(tags))
    seed_tags = sorted(x_seed[:, 0].tolist())

    if len(seed_tags) != seed_size:
        rng = random.Random(random_state)
        remaining = [tag for tag in tags if tag not in set(seed_tags)]
        if len(seed_tags) > seed_size:
            rng.shuffle(seed_tags)
            seed_tags = sorted(seed_tags[:seed_size])
        else:
            rng.shuffle(remaining)
            seed_tags = sorted(seed_tags + remaining[: seed_size - len(seed_tags)])

    seed_dataset = _subset_from_tags(dataset, seed_tags)
    x_seed_np = np.array(seed_tags, dtype=object).reshape(-1, 1)
    y_seed_np = _binarize_labels(seed_tags, seed_dataset, classes)
    _, _, x_val, _ = iterative_train_test_split(x_seed_np, y_seed_np, test_size=val_ratio)
    val_tags = sorted(x_val[:, 0].tolist())
    train_tags = sorted(tag for tag in seed_tags if tag not in set(val_tags))

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / f"seed_{seed_size}.json"
    train_path = output_dir / f"train_seed_{seed_size}.json"
    val_path = output_dir / f"val_seed_{seed_size}.json"

    save_dataset(seed_dataset, seed_path)
    save_dataset(_subset_from_tags(seed_dataset, train_tags), train_path)
    save_dataset(_subset_from_tags(seed_dataset, val_tags), val_path)

    return {"seed": seed_path, "train": train_path, "val": val_path}


class MinifiguresDataset(Dataset):
    """Notebook-style custom dataset."""

    def __init__(self, data_f: Path, dataset: dict[str, list[str]], resolution: int = 256) -> None:
        self.data_f = data_f
        self.keys, self.labels = zip(*dataset.items())
        self.classes = sorted({label for labels in self.labels for label in labels})
        self.transform_resize = get_transform_resize(resolution)
        self.transform_normalize = get_transform_normalize()

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image = load_img(self.data_f / f"{self.keys[idx]}.png")
        image = self.transform_resize(image[None,])[0]
        image = self.transform_normalize(image)
        label = torch.FloatTensor([item in self.labels[idx] for item in self.classes])
        return {"image": image, "label": label, "tag": self.keys[idx]}


def make_dataloader(
    dataset_path: Path, batch_size: int = 8, shuffle: bool = False, drop_last: bool = False
) -> DataLoader:
    """Create a notebook-style DataLoader for a JSON split."""
    dataset = MinifiguresDataset(data_f=IMAGES_DIR, dataset=load_dataset(dataset_path))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def metric_precision(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Precision is the fraction of relevant instances among retrieved ones."""
    if pred.sum() == 0:
        return 1.0
    if target.sum() == 0:
        return 0.0
    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    return float(tp / max(tp + fp, 1e-8))


def metric_recall(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Recall is the fraction of relevant instances that were retrieved."""
    if target.sum() == 0:
        return 1.0
    if pred.sum() == 0:
        return 0.0
    tp = (pred * target).sum()
    fn = (target * (1 - pred)).sum()
    return float(tp / max(tp + fn, 1e-8))


def metric_f1_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """The F1 score is the harmonic mean of precision and recall."""
    precision = metric_precision(pred=pred, target=target)
    recall = metric_recall(pred=pred, target=target)
    return 2 * (precision * recall) / max(precision + recall, 1e-8)


def metric_class_balanced_f1(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate the class-balanced F1 score from the notebook."""
    f1s = [
        metric_f1_score(
            pred=(logits[:, idx] >= 0.0).to(torch.float64),
            target=(target[:, idx] >= 0.5).to(torch.float64),
        )
        for idx in range(target.shape[1])
    ]
    return torch.mean(torch.tensor(f1s))


def train_epoch(
    dataloader: DataLoader, model: EncoderDecoder, optimizer: torch.optim.Adam, loss_fn: nn.Module
) -> tuple[list[float], list[float]]:
    """Train the model for one epoch."""
    model.train()
    losses, f1_scores = [], []
    for batch in dataloader:
        pred = model(batch["image"])
        loss = loss_fn(pred, batch["label"])
        f1 = metric_class_balanced_f1(pred, batch["label"])

        losses.append(loss.item())
        f1_scores.append(f1.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses, f1_scores


def validate_epoch(
    dataloader: DataLoader, model: EncoderDecoder, loss_fn: nn.Module
) -> dict[str, float]:
    """Validate the model on the validation set."""
    model.eval()
    test_loss, test_f1 = [], []
    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch["image"])
            test_loss += [loss_fn(pred, batch["label"]).item()] * batch["label"].shape[0]
            test_f1 += [metric_class_balanced_f1(pred, batch["label"]).item()] * batch[
                "label"
            ].shape[0]
    return {"loss": sum(test_loss) / len(test_loss), "f1": sum(test_f1) / len(test_f1)}


def finetune_model(
    base_model_tag: str = DEFAULT_BASE_MODEL_TAG,
    output_model_tag: str = DEFAULT_OUTPUT_MODEL_TAG,
    train_dataset_path: Path | None = None,
    val_dataset_path: Path | None = None,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
) -> dict[str, float]:
    """Continue training from a saved model and persist the best checkpoint."""
    train_dataset_path = train_dataset_path or get_latest_dataset_path("train_seed_*.json")
    val_dataset_path = val_dataset_path or get_latest_dataset_path("val_seed_*.json")

    train_dataset = load_dataset(train_dataset_path)
    val_dataset = load_dataset(val_dataset_path)

    dataset_train = MinifiguresDataset(data_f=IMAGES_DIR, dataset=train_dataset)
    dataset_val = MinifiguresDataset(data_f=IMAGES_DIR, dataset=val_dataset)
    assert dataset_train.classes == dataset_val.classes

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    model = EncoderDecoder.load(base_model_tag)
    model.tag = output_model_tag

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_metrics = {"loss": float("inf"), "f1": 0.0}
    for epoch in range(1, epochs + 1):
        losses_epoch, f1_epoch = train_epoch(
            dataloader=loader_train, model=model, optimizer=optimizer, loss_fn=loss_fn
        )
        metrics = validate_epoch(dataloader=loader_val, model=model, loss_fn=loss_fn)

        train_loss = sum(losses_epoch) / len(losses_epoch)
        train_f1 = sum(f1_epoch) / len(f1_epoch)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
            f"val_loss={metrics['loss']:.4f} val_f1={metrics['f1']:.4f}"
        )

        if metrics["loss"] < best_metrics["loss"]:
            best_metrics = metrics
            model.save()

    return best_metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-seed-split")
    prepare_parser.add_argument("--source-dataset-path", type=Path, default=MERGED_DATASET_PATH)
    prepare_parser.add_argument("--output-dir", type=Path, default=DATASETS_DIR)
    prepare_parser.add_argument("--seed-size", type=int, default=50)
    prepare_parser.add_argument("--val-ratio", type=float, default=0.2)
    prepare_parser.add_argument("--random-state", type=int, default=42)

    train_parser = subparsers.add_parser("train-seed-model")
    train_parser.add_argument("--base-model-tag", default=DEFAULT_BASE_MODEL_TAG)
    train_parser.add_argument("--output-model-tag", default=DEFAULT_OUTPUT_MODEL_TAG)
    train_parser.add_argument("--train-dataset-path", type=Path)
    train_parser.add_argument("--val-dataset-path", type=Path)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)

    return parser.parse_args()


def main() -> None:
    """Run the selected finetuning task."""
    args = parse_args()

    if args.command == "prepare-seed-split":
        paths = create_seed_splits(
            source_dataset_path=args.source_dataset_path,
            output_dir=args.output_dir,
            seed_size=args.seed_size,
            val_ratio=args.val_ratio,
            random_state=args.random_state,
        )
        for name, path in paths.items():
            print(f"{name}_path: {path}")
        return

    if args.command == "train-seed-model":
        train_dataset_path = args.train_dataset_path or get_latest_dataset_path("train_seed_*.json")
        val_dataset_path = args.val_dataset_path or get_latest_dataset_path("val_seed_*.json")
        print(f"train_dataset_path: {train_dataset_path}")
        print(f"val_dataset_path: {val_dataset_path}")
        metrics = finetune_model(
            base_model_tag=args.base_model_tag,
            output_model_tag=args.output_model_tag,
            train_dataset_path=train_dataset_path,
            val_dataset_path=val_dataset_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        print(f"best_val_loss: {metrics['loss']:.4f}")
        print(f"best_val_f1: {metrics['f1']:.4f}")
        return

    msg = f"Unsupported command: {args.command}"
    raise ValueError(msg)


if __name__ == "__main__":
    main()
