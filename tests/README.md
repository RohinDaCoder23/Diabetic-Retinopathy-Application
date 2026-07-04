# Tests

Two ways to verify the project, depending on what's installed.

## 1. Zero-dependency offline check (no pip install needed)

Runs the image-processing core (preprocess, synthetic data, config, the CNN
parameter-count math) using only numpy + OpenCV + pandas + PyYAML. It auto-runs
the heavier torch/sklearn checks too, *if* those happen to be installed.

```bash
python tests/run_offline_checks.py
```

Exit code 0 means every check that could run passed. Missing optional deps show
as `SKIP`, not failure.

## 2. Full pytest suite

Needs the project deps plus pytest:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest                     # everything
pytest -m "not slow"       # skip the end-to-end train/evaluate tests (fast)
pytest -m slow             # ONLY the end-to-end pipeline tests
pytest tests/test_preprocess.py -v   # one file
```

### What's covered

| File                     | What it checks                                            | Needs |
|--------------------------|-----------------------------------------------------------|-------|
| `test_config.py`         | `config.yaml` is valid + has all required sections        | pyyaml |
| `test_preprocess.py`     | FOV crop, Ben-Graham, CLAHE, resize, determinism          | opencv |
| `test_data.py`           | synthetic gen, label reading, stratified split, weights   | (sklearn/torch for some) |
| `test_augment.py`        | train/eval transforms output CHW tensors; eval is stable  | albumentations, torch |
| `test_models.py`         | every backbone builds + forwards to `(batch, 5)`; param count; freeze | torch |
| `test_end_to_end.py`     | train → checkpoint → evaluate → inference → Grad-CAM       | torch, sklearn, albumentations |

Tests that need a missing library **skip** rather than fail, so the suite is
safe to run in a partial environment. Transfer models are built with
`pretrained=False`, so tests never download ImageNet weights or hit the network.

All tests use a tiny **synthetic** dataset generated on the fly — you do NOT need
to download APTOS to run the tests. (Synthetic-data results are meaningless as
science; the tests only prove the code executes correctly.)
