"""Minifigures Webshop REST API."""

import logging
from importlib.metadata import version

import coloredlogs
from fastapi import FastAPI

from minifigures_api.routers import predict_router

app = FastAPI(
    title="Minifigures Webshop API",
    description="The goal of this project is to teach the students everything they need to develop and deploy an AI solution from start to finish.",
    version=version("minifigures-app"),
    docs_url="/",  # Put docs under default URL
)


@app.on_event("startup")
def startup_event() -> None:
    """Run API startup events."""
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    # Add coloredlogs' coloured StreamHandler to the root logger.
    coloredlogs.install()


# Specify the different endpoint routers
app.include_router(predict_router, prefix="/predict")
