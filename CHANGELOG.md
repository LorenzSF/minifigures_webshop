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
    ○ Pull and unzip minifigures.tar.gz
    ○ Pull dataset.json
    ○ Data should be in the ./data/data folder1
2. Have a data exploration notebook
    ○ Manually go over the data first
    ○ Analyse the data using the FastDup2 library after

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


