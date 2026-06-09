# Minifigures Webshop

## Repository Objective

This repository is the product workspace for an end-to-end AI webshop focused on LEGO minifigures.

As stated in the opening part of the course material, this repository is meant to evolve over time. It starts as the implementation workspace for the project and is expected to become the finished product repository by the end of the course.

The objective is to deliver the core building blocks of a usable AI product in a single repository:

- a dataset and training workflow for multi-label image classification
- a reusable Python model package
- a REST API for predictions and data access
- a Streamlit web interface for product browsing, prediction display, and face-based search
- tooling to label data and continuously improve the model
- infrastructure-as-code and CI/CD pipelines for cloud deployment

## Current Purpose

At the current stage, the repository already contains the main product foundations:

- a `minifigures_model` package with the classification model, fine-tuning utilities, and a face-cropper / face-similarity / face-index pipeline for face search
- a `minifigures_api` FastAPI application exposing data, prediction, and face-search endpoints
- a `minifigures_app` Streamlit application with a home page, a product page, a marketplace page, and a "Find your LEGO clone" face-search page (upload an image or use a webcam)
- labeling tooling under `src/labeling/` (Label Studio integration, active learning, annotation/embedding export and merge scripts) plus alternative Prodigy recipes under `src/prodigy_recipes/`
- notebooks for data exploration, dataset preparation, and model fine-tuning
- local dataset and model assets under `data/`
- cloud infrastructure provisioning via Terraform for AWS deployment (EC2, ECR, S3, Route53)
- Docker containerization and GitHub Actions pipelines for CI on pull requests and automated deployment from `main`

In practice, this means the repository already supports a complete product flow from local development to cloud deployment: frontend, backend, data and labeling tooling, and infrastructure-as-code with automated delivery.

## Current Structure

The main folders are:

- `src/minifigures_model/`: model definition, fine-tuning, and face search (cropper, embeddings, index) utilities
- `src/minifigures_api/`: backend service and API routers (`data`, `predict`, `face_search`)
- `src/minifigures_app/`: customer-facing web interface built with Streamlit (Home, Product, Market, Find your LEGO clone)
- `src/labeling/`: Label Studio configuration, active learning, and annotation/embedding export and merge scripts
- `src/prodigy_recipes/`: alternative Prodigy recipes for data labeling
- `src/cicd-example/`: course example illustrating a CI/CD pipeline with GitHub Actions and Docker
- `data/`: dataset files, split files, notebooks, and local product/model assets
- `tests/`: unit and API tests covering the model, face search, and app utilities
- `infrastructure/terraform/`: Infrastructure as Code for AWS cloud deployment
  - `modules/student-stack/`: Terraform module for EC2, ECR, S3, and Route53 provisioning
- `shared-resources/`: course-provided supporting material with shared Terraform modules (Git submodule)
- `validity_predict_endpoint/`: helper script to validate the response contract of a deployed prediction endpoint
- `.github/workflows/`: CI pipeline for pull requests and deployment pipeline for the `main` branch

## Environment And Commands

This project uses `uv` for dependency management and `Poe the Poet` for common project commands.

Useful commands:

- `uv sync --all-groups` installs the project dependencies
- `uv run poe api` runs the REST API
- `uv run poe app` runs the Streamlit application
- `uv run poe lint` runs linting and formatting hooks
- `uv run poe test` runs the test suite

If the minifigures dataset is not available locally, run:

- `./setup_data.sh`

## Open The Provisional Web Application

To open the current provisional version of the product interface in the available viewer, use two terminals from the repository root.

Terminal 1:

```bash
uv sync --all-groups
uv run poe api --dev --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
uv run poe app --host 0.0.0.0 --port 8500
```

No additional `API_HOST` configuration is required for local runs. By default, the app connects to the FastAPI backend at `http://localhost:8000`.

Then open the forwarded port for `8500` in the VS Code port viewer or integrated browser. The backend API documentation is available at `http://localhost:8000/docs`.

## Current Limitation

The interface can already be opened as a provisional product demo, but full prediction and face-search support depends on having the corresponding trained model artifacts (classification model, face cropper, and face embedding index) saved in the expected local model folder. Until those artifacts are generated and saved, the interface can be explored but prediction and face-search features may remain incomplete.
