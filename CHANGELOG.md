# Project Log - Minifigures Webshop

## General Information
- **Project:** Minifigures Webshop
- **Repository:** kuleuven-realization-of-ai-classroom-2026-updated-minifigures-webshop-2026-minifigures-webshop-templ
- **Current Branch:** main

---

## Change Log

### [SPRINT 1]
#### objectives
1. Extend the pyproject.toml file with a Poe the Poet task to pull the provided data
  - Pull and unzip minifigures.tar.gz
  - Pull dataset.json
  - Data should be in the ./data/data folder1
2. Have a data exploration notebook
  - Manually go over the data first
  - Analyse the data using the FastDup2 library after

#### Additional changes made 
- Added `CHANGELOG.md` file to track project milestones and changes
- Integrated `shared-resources` repository as Git submodule for automatic updates


#### Additional Notes
- To update shared resources: `git submodule update --remote`
- **Git**: Version control and repository management
  - Submodule configuration for shared-resources integration
  - Branch synchronization and push operations
- **DVC (Data Version Control)**: Dataset management
    - Dataset stored in remote S3: `s3://roai-data-readonly` (eu-central-1)
- **curl/wget**: File download and API interactions
- **tar**: Archive extraction (`tar -xzf`)


### [SPRINT 2]
#### objectives
1. Create Python scripts (in minifigures_model folder)
  - Model and dataset creation
  - Trainer class to train and validate your model
  - Evaluation function to evaluate a provided model
  - Add documentation in your README.md
2. Wrap a pre-trained model in a REST API (in minifigures_api folder)
  - Update the predict endpoint (class probabilities)
  - Create data endpoints to list all image tags and fetch a specific image
3. Visualise predictions in Streamlit (in minifigures_app folder)
  - Show the predict results from an uploaded or randomly sampled file
  - Create the marketplace with pagination
  - Open Product tab from marketplace

