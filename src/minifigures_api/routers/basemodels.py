"""Pydantic basemodels."""

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """Prediction base model."""

    prediction: dict[str, float] = Field(
        ...,
        title="prediction",
        description="Dictionary containing the predicted probability for each class",
        example={"class1": 0.42, "class2": 0.42},
    )


class FaceBox(BaseModel):
    """Normalized face box response model."""

    x_center: float = Field(..., ge=0, le=1)
    y_center: float = Field(..., ge=0, le=1)
    width: float = Field(..., gt=0, le=1)
    height: float = Field(..., gt=0, le=1)


class FaceMatch(BaseModel):
    """Face similarity match response model."""

    tag: str = Field(..., title="tag", description="Catalog image tag")
    score: float = Field(..., title="score", description="Cosine similarity score")


class FaceSearchResponse(BaseModel):
    """Face search response model."""

    query_box: FaceBox = Field(..., title="query_box")
    matches: list[FaceMatch] = Field(..., title="matches")
