"""Shared Label Studio upload helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from label_studio_sdk import Client

try:
    from config import HOST, PORT
except ModuleNotFoundError:
    from src.labeling.config import HOST, PORT


def get_label_studio_project(token: str, project_id: int):
    """Load the target Label Studio project."""
    ls = Client(url=f"{HOST}:{PORT}", api_key=token)
    return ls.get_project(project_id)


def assert_new_model_version(project, model_version: str) -> None:
    """Make sure the model version does not already exist."""
    assert model_version not in project.get_model_versions(), (
        "The model_version you provided already exists. Choose a different one."
    )


def get_task_tag(task: dict[str, Any]) -> str:
    """Extract the minifigure tag from a Label Studio task."""
    return Path(task["storage_filename"]).stem


def build_choices_result(choices: list[str]) -> list[dict[str, Any]]:
    """Build a Label Studio choices prediction payload."""
    if not choices:
        return []
    return [
        {
            "from_name": "choice",
            "to_name": "image",
            "type": "choices",
            "value": {"choices": choices},
        }
    ]


def add_predictions_to_label_studio(
    tags_payloads: dict[str, dict[str, Any]], model_version: str, token: str, project_id: int
) -> None:
    """Upload prediction payloads to Label Studio tasks."""
    project = get_label_studio_project(token=token, project_id=project_id)
    assert_new_model_version(project=project, model_version=model_version)

    print("Setting predictions in label-studio...")
    created_count = 0

    # Iterate over Label Studio tasks.
    for task in project.get_tasks():
        tag = get_task_tag(task)
        payload = tags_payloads.get(tag)
        if payload is None:
            continue
        if payload["score"] <= 0.0 and not payload["result"]:
            continue

        project.create_prediction(
            task["id"],
            result=payload["result"],
            score=payload["score"],
            model_version=model_version,
        )
        created_count += 1

    print(f"created_count: {created_count}")


def add_prediction_scores_to_label_studio(
    tags_scores: dict[str, float], model_version: str, token: str, project_id: int
) -> None:
    """Add prediction score attributes to Label Studio tasks."""
    add_predictions_to_label_studio(
        tags_payloads={tag: {"result": [], "score": score} for tag, score in tags_scores.items()},
        model_version=model_version,
        token=token,
        project_id=project_id,
    )
