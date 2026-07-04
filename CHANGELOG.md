# Changelog

Notable changes to this project, newest first.

## [0.2.0] — Testing, CI & one-click runner
- Added a pytest suite under `tests/` covering config validation, preprocessing,
  data loading and stratified splits, augmentation, model builders, and an
  end-to-end train → evaluate → inference → Grad-CAM test. Tests run on a small
  synthetic dataset with `pretrained=False`, so no data download or network
  access is required.
- Added `tests/run_offline_checks.py`, a zero-dependency smoke check for
  environments without the full ML stack installed.
- Added `pytest.ini` and `requirements-dev.txt`.
- Added `.github/workflows/ci.yml` — runs the offline checks and the full test
  suite on every push and pull request.
- Added `CONTRIBUTING.md` and `notebooks/run_all_colab.ipynb` (a one-click Colab
  driver: install → self-test → download data → train → evaluate → explain).
- README: added CI/license badges and a Testing section.

## [0.1.0] — Initial project

### Data & EDA
- `src/data.py`: label reading, stratified train/val/test split, class weights,
  weighted sampler, `APTOSDataset`, and dataloaders.
- `notebooks/01_eda.ipynb`: class distribution, sample grid, quality checks.

### Preprocessing & augmentation
- `src/preprocess.py`: field-of-view crop, Ben-Graham normalization, CLAHE, resize.
- `src/augment.py`: gentle Albumentations train transforms, eval transforms,
  denormalize.
- `notebooks/02_preprocessing.ipynb`: before/after visualizations.

### Models
- `src/models/custom_cnn.py`: a from-scratch 4-block CNN (~390,181 parameters).
- `src/models/transfer.py`: ResNet50, EfficientNet-B0/B3, DenseNet121 builders with
  ImageNet weights and swapped 5-class heads; freeze/unfreeze helper.
- `src/models/__init__.py`: `build_model(cfg)` factory and Grad-CAM target lookup.

### Training & evaluation
- `src/train.py`: class-weighted cross-entropy / focal loss, AdamW, cosine/plateau
  schedulers, mixed precision, gradient clipping, early stopping on QWK,
  checkpointing, and a `--smoke-test` mode.
- `src/evaluate.py`: accuracy, per-class and macro precision/recall/F1, macro
  ROC-AUC, confusion matrix, Quadratic Weighted Kappa, and referable-DR recall.
- `notebooks/03_results.ipynb`: comparison table, confusion grids, ROC curves.

### Explainability & app
- `src/gradcam.py`: Grad-CAM heatmap gallery and single-image helper.
- `app/streamlit_app.py`: upload a fundus image, pick a model, and view the
  predicted grade, probability chart, and Grad-CAM overlay.
- `src/inference.py`: single-image inference sharing the training-time preprocessing.

### Docs
- `docs/concepts.md`, `docs/pipeline.md`, `docs/colab_quickstart.md`,
  `docs/ethics_limitations.md`, `docs/presentation_notes.md`, and the README.

> Note: `data/_synthetic_demo/` may appear from figure generation; it is
> gitignored (everything under `data/`) and safe to delete.
