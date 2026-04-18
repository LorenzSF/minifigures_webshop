"""Export model probabilities for all images to a JSON file."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from minifigures_model.data_utils import ALL_PREDICTIONS_PATH, list_image_paths, write_json
from minifigures_model.model import EncoderDecoder
from minifigures_model.preprocessing import preprocess_pil_image

INFERENCE_BATCH_SIZE = 32  # Size needed to process with CPU.


def predict_batch(model: EncoderDecoder, image_paths: list[Path]) -> dict[str, dict[str, float]]:
    """Return class probabilities for one batch of images."""
    images_t = []

    # Load and preprocess a batch of images.
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images_t.append(preprocess_pil_image(image, resolution=model.resolution))

    batch_t = torch.stack(images_t)
    logits = model(batch_t)
    probs = torch.sigmoid(logits).detach().cpu().tolist()

    predictions = {}

    # Convert batch outputs into a tag -> class score mapping.
    for image_path, image_probs in zip(image_paths, probs, strict=True):
        predictions[image_path.stem] = {
            name: float(score) for name, score in zip(model.classes, image_probs, strict=True)
        }
    return predictions


def export_predictions(model_tag: str, output_path: Path = ALL_PREDICTIONS_PATH) -> Path:
    """Export probabilities for every image."""
    model = EncoderDecoder.load(model_tag)
    model.eval()

    image_paths = list_image_paths()
    predictions = {}
    print(f"images_total: {len(image_paths)}")
    print(f"inference_batch_size: {INFERENCE_BATCH_SIZE}")

    with torch.inference_mode():
        # Run inference in fixed-size batches.
        for start_idx in range(0, len(image_paths), INFERENCE_BATCH_SIZE):
            batch_paths = image_paths[start_idx : start_idx + INFERENCE_BATCH_SIZE]
            predictions.update(predict_batch(model, batch_paths))
            idx = start_idx + len(batch_paths)
            if idx % 250 == 0 or idx == len(image_paths):
                print(f"processed_count: {idx}")

    payload = {"model_tag": model_tag, "classes": model.classes, "predictions": predictions}

    write_json(output_path, payload, sort_keys=True)

    print(f"model_tag: {model_tag}")
    print(f"output_path: {output_path}")
    print(f"image_count: {len(predictions)}")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--output-path", type=Path, default=ALL_PREDICTIONS_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the export command."""
    args = parse_args()
    export_predictions(model_tag=args.model_tag, output_path=args.output_path)


if __name__ == "__main__":
    main()
