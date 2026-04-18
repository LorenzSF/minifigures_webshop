## Label Studio And Active Learning Workflow

This workflow uses Label Studio for manual annotation and the repo scripts for:

- exporting labels
- merging labels into the working dataset
- preparing train/val splits
- fine-tuning the model
- ranking unlabeled tasks with active learning
- uploading selected model predictions for review

## 1. Start Label Studio

From the repository root:

```bash
export PYTHONPATH=src
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/workspaces
label-studio
```

Notes:

- Use the browser inside the online VS Code / codespace. The local VS Code browser integration often fails.
- If the UI hangs, refresh the page or restart the server.
- Set the two `LABEL_STUDIO_*` variables every time you start the server.

## 2. Create The Project And Storages

In Label Studio:

1. Create a new project.
2. Paste `src/labeling/project_config.xml` into the labeling interface.
3. Add a source storage:
   `data/data/minifigures`
4. Set the file filter regex:
   `^(?!\._).*\.png$`
5. Enable the option that treats each file as one task.
6. Sync the storage.
7. Add a target storage:
   `data/data/target_annotations`

Notes:

- If sync raises a runtime error, run sync again. Partial progress is usually kept.
- Label Studio only manages labeled vs unlabeled tasks. Train/val splitting is handled by the repo scripts.

## 3. First Labeling Round

For the first round, use random sampling:

1. Open `Settings > General > Task Sampling`
2. Select `Random sampling`
3. Save
4. Start labeling

## 4. Export And Merge Labels

After each labeling round, from the repository root:

```bash
export PYTHONPATH=src
python src/labeling/export_annotations.py
python src/labeling/compare_annotations.py
python src/labeling/merge_annotations.py
```

Files used now:

- `data/data/dataset_labeled.json`
- `data/data/dataset_merged_labeled.json`

Behavior:

- `export_annotations.py` overwrites `dataset_labeled.json`
- `compare_annotations.py` compares `dataset.json` vs `dataset_labeled.json`
- `merge_annotations.py` merges new tags into `dataset_merged_labeled.json`

## 5. Prepare The Seed Split

From the repository root:

```bash
export PYTHONPATH=src
python -m minifigures_model.model_finetune prepare-seed-split --seed-size 300
```

This writes:

- `data/data/datasets/seed_300.json`
- `data/data/datasets/train_seed_300.json`
- `data/data/datasets/val_seed_300.json`

## 6. Fine-Tune The Model

Example:

```bash
export PYTHONPATH=src
python -m minifigures_model.model_finetune train-seed-model \
  --base-model-tag my_model_active_learning_v5 \
  --output-model-tag my_model_active_learning_v6 \
  --epochs 10
```

Notes:

- The command automatically picks the latest `train_seed_*.json` and `val_seed_*.json`
- For the first run, `--base-model-tag my_model` is a valid starting point
- The best checkpoint is saved under `data/models/<output-model-tag>/`

## 7. Run Active Learning

This script only does `loss + KNN` ranking for unlabeled tasks.

Example:

```bash
export PYTHONPATH=src
python src/labeling/active_learning.py \
  --model-version active_seed500_v1 \
  --token YOUR_LABEL_STUDIO_TOKEN \
  --project-id 1 \
  --model-tag my_model_active_learning_v6 \
  --budget 100 \
  --k 5
```

Notes:

- `--budget` controls how many unlabeled tasks receive a score
- `--k` controls how many nearest neighbors are considered per difficult train sample
- The script automatically uses the latest `train_seed_*.json` unless you pass `--train-dataset-path`

## 8. Review The Prioritized Tasks In Label Studio

After uploading active learning scores:

1. Open `Settings > Machine Learning`
2. Select the `Model Version` used in the command
3. Enable the `Prediction score` column
4. Sort by `Prediction score`
5. Use `Label Tasks As Displayed`

## 9. Export Predictions For Review

If you want to review model predictions directly, first export them:

```bash
export PYTHONPATH=src
python src/labeling/export_predictions.py \
  --model-tag my_model_active_learning_v6
```

This overwrites:

- `data/data/predictions/all_predictions.json`

You can also choose another output path:

```bash
export PYTHONPATH=src
python src/labeling/export_predictions.py \
  --model-tag my_model_active_learning_v6 \
  --output-path data/data/predictions/review_humans.json
```

## 10. Upload Predictions To Label Studio

### Single-class example

Review only `human` predictions:

```bash
export PYTHONPATH=src
python src/labeling/upload_predictions.py \
  --model-version human_v1 \
  --token YOUR_LABEL_STUDIO_TOKEN \
  --project-id 1 \
  --predictions-path data/data/predictions/all_predictions.json \
  --show-classes human \
  --score-classes human \
  --class-threshold human=0.7
```

### Multi-option example

Review several classes with different thresholds:

```bash
export PYTHONPATH=src
python src/labeling/upload_predictions.py \
  --model-version review_features_v1 \
  --token YOUR_LABEL_STUDIO_TOKEN \
  --project-id 1 \
  --predictions-path data/data/predictions/all_predictions.json \
  --show-classes human helmet cape \
  --score-classes human helmet \
  --class-threshold human=0.7 \
  --class-threshold helmet=0.8 \
  --class-threshold cape=0.6
```

Meaning:

- `--show-classes` decides which classes appear as suggested choices
- `--score-classes` decides which classes contribute to the final prediction score
- `--class-threshold class=value` can be repeated
- `alien`, `human`, and `robot` are treated as an argmax group by default
- tasks with no visible classes and score `0` are skipped

After upload:

1. Open `Settings > Machine Learning`
2. Select the uploaded `Model Version`
3. Enable the `Prediction score` column
4. Sort by `Prediction score` if needed

## 11. Repeat The Loop

Typical iteration:

1. Label tasks in Label Studio
2. Export and merge labels
3. Prepare a new seed split if needed
4. Fine-tune a new model
5. Run active learning again
6. Optionally export and upload predictions for review
