"""Train the minifigure face cropper model."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from minifigures_model.data_utils import IMAGES_DIR
from minifigures_model.face_cropper import (
    DEFAULT_FACE_CROPPER_TAG,
    FaceBox,
    FaceCropper,
    load_face_boxes,
)
from minifigures_model.preprocessing import preprocess_pil_image

DEFAULT_FACE_BOXES_PATH = IMAGES_DIR.parent / "face_boxes.json"
MIN_SPLIT_IMAGE_COUNT = 2


class FaceBoxDataset(Dataset):
    """Dataset of minifigure images and normalized face boxes."""

    def __init__(
        self,
        image_dir: Path,
        face_boxes: dict[str, FaceBox],
        image_tags: list[str],
        resolution: int,
    ) -> None:
        self.image_dir = image_dir
        self.face_boxes = face_boxes
        self.image_tags = image_tags
        self.resolution = resolution

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_tags)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one training example."""
        image_tag = self.image_tags[idx]
        image_path = self.image_dir / f"{image_tag}.png"
        with Image.open(image_path) as image:
            image_t = preprocess_pil_image(image, resolution=self.resolution)
        return {"image": image_t, "box": self.face_boxes[image_tag].to_tensor(), "tag": image_tag}


def split_tags(
    image_tags: list[str], *, val_ratio: float, random_state: int
) -> tuple[list[str], list[str]]:
    """Create a deterministic train/validation split."""
    if not 0 < val_ratio < 1:
        msg = f"val_ratio must be within (0, 1), got {val_ratio}."
        raise ValueError(msg)
    if len(image_tags) < MIN_SPLIT_IMAGE_COUNT:
        msg = "At least two annotated images are required for training."
        raise ValueError(msg)

    shuffled_tags = list(image_tags)
    random.Random(random_state).shuffle(shuffled_tags)  # noqa: S311
    val_size = min(max(1, round(len(shuffled_tags) * val_ratio)), len(shuffled_tags) - 1)
    return sorted(shuffled_tags[val_size:]), sorted(shuffled_tags[:val_size])


def get_annotated_image_tags(image_dir: Path, face_boxes: dict[str, FaceBox]) -> list[str]:
    """Return annotated tags that have matching catalog images."""
    tags = sorted(tag for tag in face_boxes if (image_dir / f"{tag}.png").is_file())
    if not tags:
        msg = f"No annotated image tags with files were found in {image_dir}."
        raise FileNotFoundError(msg)
    return tags


def train_epoch(
    dataloader: DataLoader, model: FaceCropper, optimizer: torch.optim.Optimizer, loss_fn: nn.Module
) -> float:
    """Train one cropper epoch."""
    model.train()
    losses = []
    for batch in dataloader:
        prediction = model(batch["image"])
        loss = loss_fn(prediction, batch["box"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def validate_epoch(dataloader: DataLoader, model: FaceCropper, loss_fn: nn.Module) -> float:
    """Validate the cropper for one epoch."""
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in dataloader:
            prediction = model(batch["image"])
            losses.append(float(loss_fn(prediction, batch["box"]).item()))
    return sum(losses) / len(losses)


def train_face_cropper(  # noqa: PLR0913
    face_boxes_path: Path = DEFAULT_FACE_BOXES_PATH,
    image_dir: Path = IMAGES_DIR,
    output_tag: str = DEFAULT_FACE_CROPPER_TAG,
    *,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.2,
    random_state: int = 42,
    use_pretrained_encoder: bool = False,
) -> dict[str, float]:
    """Train and save the best face cropper checkpoint."""
    face_boxes = load_face_boxes(face_boxes_path)
    image_tags = get_annotated_image_tags(image_dir, face_boxes)
    train_tags, val_tags = split_tags(image_tags, val_ratio=val_ratio, random_state=random_state)

    model = FaceCropper(tag=output_tag, pretrained=use_pretrained_encoder)
    train_dataset = FaceBoxDataset(image_dir, face_boxes, train_tags, model.resolution)
    val_dataset = FaceBoxDataset(image_dir, face_boxes, val_tags, model.resolution)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    best_metrics = {"loss": float("inf")}

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(train_loader, model, optimizer, loss_fn)
        val_loss = validate_epoch(val_loader, model, loss_fn)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_metrics["loss"]:
            best_metrics = {"loss": val_loss}
            model.save()

    return best_metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-boxes-path", type=Path, default=DEFAULT_FACE_BOXES_PATH)
    parser.add_argument("--image-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--output-tag", default=DEFAULT_FACE_CROPPER_TAG)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-pretrained-encoder", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run face cropper training."""
    args = parse_args()
    metrics = train_face_cropper(
        face_boxes_path=args.face_boxes_path,
        image_dir=args.image_dir,
        output_tag=args.output_tag,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
        use_pretrained_encoder=args.use_pretrained_encoder,
    )
    print(f"best_val_loss: {metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
