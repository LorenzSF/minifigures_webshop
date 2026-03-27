# Minifigures Webshop

## Repository Objective

This repository is the product workspace for an end-to-end AI webshop focused on LEGO minifigures.

As stated in the opening part of the course material, this repository is meant to evolve over time. It starts as the implementation workspace for the project and is expected to become the finished product repository by the end of the course.

The objective is to deliver the core building blocks of a usable AI product in a single repository:

- a dataset and training workflow for multi-label image classification
- a reusable Python model package
- a REST API for predictions and data access
- a Streamlit web interface for product browsing and prediction display

## Current Purpose

At the current stage, the repository already contains the main product foundations:

- a `minifigures_model` package with model and utility code
- a `minifigures_api` FastAPI application exposing data and prediction endpoints
- a `minifigures_app` Streamlit application with a home page, product page, and marketplace page
- notebooks for dataset preparation, custom datasets, and model fine-tuning
- local dataset assets under `data/`
- cloud infrastructure provisioning via Terraform for AWS deployment (EC2, ECR, S3, Route53)
- Docker containerization for application deployment on AWS EC2

In practice, this means the repository already supports a complete product flow from local development to cloud deployment: frontend, backend, data assets, and infrastructure-as-code for production readiness.

## Current Structure

The main folders are:

- `src/minifigures_model/`: model definition and model-related utilities
- `src/minifigures_api/`: backend service and API routers
- `src/minifigures_app/`: provisional customer-facing web interface built with Streamlit
- `data/`: dataset files, split files, notebooks, and local product assets
- `tests/`: import and API tests
- `infrastructure/terraform/`: Infrastructure as Code for AWS cloud deployment
  - `modules/student-stack/`: Terraform module for EC2, ECR, S3, and Route53 provisioning
- `shared-resources/`: course-provided supporting material with shared Terraform modules

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

Then open the forwarded port for `8500` in the VS Code port viewer or integrated browser. The backend API documentation is available on port `8000`.

## Current Limitation

The interface can already be opened as a provisional product demo, but full prediction support depends on having a trained model saved in the expected local model folder. Until that model artifact is generated and saved, the interface can be explored but prediction features may remain incomplete.

