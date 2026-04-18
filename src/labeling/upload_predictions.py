"""Upload exported model predictions to Label Studio."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from label_studio_utils import add_predictions_to_label_studio, build_choices_result
except ModuleNotFoundError:
    from src.labeling.label_studio_utils import (
        add_predictions_to_label_studio,
        build_choices_result,
    )

from minifigures_model.data_utils import ALL_PREDICTIONS_PATH, load_json

DEFAULT_CLASS_THRESHOLD = 0.8
DEFAULT_ARGMAX_CLASSES = ["alien", "human", "robot"]


def load_predictions_index(predictions_path: Path) -> dict[str, dict[str, float]]:
    """Load the exported predictions from disk."""
    payload = load_json(predictions_path)
    predictions = payload.get("predictions", {})
    if not predictions:
        raise ValueError(f"No predictions found in {predictions_path}.")
    return predictions


def unique_classes(classes: list[str] | None) -> list[str]:
    """Remove duplicates while preserving order."""
    if not classes:
        return []
    return list(dict.fromkeys(classes))


def parse_class_thresholds(items: list[str] | None) -> dict[str, float]:
    """Parse class thresholds from repeated class=value arguments."""
    thresholds = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid class threshold '{item}'. Use class=value.")
        class_name, value = item.split("=", 1)
        thresholds[class_name] = float(value)
    return thresholds


def validate_classes(class_names: list[str], available_classes: list[str], label: str) -> None:
    """Validate that all requested classes exist in the predictions index."""
    invalid = sorted(set(class_names) - set(available_classes))
    if invalid:
        raise ValueError(
            f"Unknown {label}: {invalid}. Available classes: {sorted(available_classes)}"
        )


def get_class_threshold(class_name: str, class_thresholds: dict[str, float]) -> float:
    """Return the threshold for a class, or the default one."""
    return class_thresholds.get(class_name, DEFAULT_CLASS_THRESHOLD)


def get_argmax_prediction(
    class_scores: dict[str, float], argmax_classes: list[str]
) -> tuple[str, float] | None:
    """Return the winning class inside the mutually exclusive group."""
    if not argmax_classes:
        return None
    winner = max(argmax_classes, key=lambda class_name: class_scores[class_name])
    return winner, float(class_scores[winner])


def get_visible_prediction_classes(
    class_scores: dict[str, float],
    show_classes: list[str],
    argmax_classes: list[str],
    class_thresholds: dict[str, float],
) -> list[str]:
    """Return the classes that should be visible in Label Studio."""
    visible = []
    argmax_result = get_argmax_prediction(class_scores=class_scores, argmax_classes=argmax_classes)
    argmax_set = set(argmax_classes)

    if argmax_result is not None:
        winner, winner_score = argmax_result
        if winner in show_classes and winner_score >= get_class_threshold(winner, class_thresholds):
            visible.append(winner)

    # Keep the remaining classes independent from the argmax group.
    for class_name in show_classes:
        if class_name in argmax_set:
            continue
        if class_scores[class_name] >= get_class_threshold(class_name, class_thresholds):
            visible.append(class_name)

    return visible


def get_prediction_score(
    class_scores: dict[str, float],
    score_classes: list[str],
    argmax_classes: list[str],
    class_thresholds: dict[str, float],
) -> float:
    """Aggregate the selected classes into a final Label Studio score."""
    if not score_classes:
        return 0.0

    eligible_scores = []
    argmax_result = get_argmax_prediction(class_scores=class_scores, argmax_classes=argmax_classes)
    argmax_set = set(argmax_classes)

    if argmax_result is not None:
        winner, winner_score = argmax_result
        if winner in score_classes and winner_score >= get_class_threshold(
            winner, class_thresholds
        ):
            eligible_scores.append(winner_score)

    # Keep the remaining classes independent from the argmax group.
    for class_name in score_classes:
        if class_name in argmax_set:
            continue
        class_score = float(class_scores[class_name])
        if class_score >= get_class_threshold(class_name, class_thresholds):
            eligible_scores.append(class_score)

    return max(eligible_scores, default=0.0)


def build_prediction_payloads(
    predictions_index: dict[str, dict[str, float]],
    show_classes: list[str],
    score_classes: list[str],
    argmax_classes: list[str],
    class_thresholds: dict[str, float],
) -> dict[str, dict[str, float | list[dict[str, object]]]]:
    """Build Label Studio payloads from exported predictions."""
    payloads = {}

    # Build one payload per image tag.
    for tag, class_scores in predictions_index.items():
        visible_classes = get_visible_prediction_classes(
            class_scores=class_scores,
            show_classes=show_classes,
            argmax_classes=argmax_classes,
            class_thresholds=class_thresholds,
        )
        payloads[tag] = {
            "result": build_choices_result(visible_classes),
            "score": get_prediction_score(
                class_scores=class_scores,
                score_classes=score_classes,
                argmax_classes=argmax_classes,
                class_thresholds=class_thresholds,
            ),
        }

    return payloads


def upload_predictions(
    model_version: str,
    token: str,
    project_id: int,
    predictions_path: Path = ALL_PREDICTIONS_PATH,
    show_classes: list[str] | None = None,
    score_classes: list[str] | None = None,
    argmax_classes: list[str] | None = None,
    class_thresholds: dict[str, float] | None = None,
) -> None:
    """Upload model predictions from JSON to Label Studio."""
    predictions_index = load_predictions_index(predictions_path=predictions_path)
    available_classes = list(next(iter(predictions_index.values())).keys())

    show_classes = unique_classes(show_classes)
    argmax_classes = unique_classes(argmax_classes) or DEFAULT_ARGMAX_CLASSES
    score_classes = unique_classes(score_classes) or show_classes
    class_thresholds = class_thresholds or {}

    validate_classes(show_classes, available_classes=available_classes, label="show_classes")
    validate_classes(score_classes, available_classes=available_classes, label="score_classes")
    validate_classes(argmax_classes, available_classes=available_classes, label="argmax_classes")
    validate_classes(
        list(class_thresholds), available_classes=available_classes, label="class_thresholds"
    )

    payloads = build_prediction_payloads(
        predictions_index=predictions_index,
        show_classes=show_classes,
        score_classes=score_classes,
        argmax_classes=argmax_classes,
        class_thresholds=class_thresholds,
    )

    print(f"predictions_path: {predictions_path}")
    print(f"show_classes: {show_classes}")
    print(f"score_classes: {score_classes}")
    print(f"argmax_classes: {argmax_classes}")
    print(f"class_thresholds: {class_thresholds}")
    add_predictions_to_label_studio(
        tags_payloads=payloads, model_version=model_version, token=token, project_id=project_id
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--project-id", required=True, type=int)
    parser.add_argument("--predictions-path", type=Path, default=ALL_PREDICTIONS_PATH)
    parser.add_argument("--show-classes", nargs="*")
    parser.add_argument("--score-classes", nargs="*")
    parser.add_argument("--argmax-classes", nargs="*", default=DEFAULT_ARGMAX_CLASSES)
    parser.add_argument("--class-threshold", action="append")
    return parser.parse_args()


def main() -> None:
    """Run the upload command."""
    args = parse_args()
    upload_predictions(
        model_version=args.model_version,
        token=args.token,
        project_id=args.project_id,
        predictions_path=args.predictions_path,
        show_classes=args.show_classes,
        score_classes=args.score_classes,
        argmax_classes=args.argmax_classes,
        class_thresholds=parse_class_thresholds(args.class_threshold),
    )


if __name__ == "__main__":
    main()
