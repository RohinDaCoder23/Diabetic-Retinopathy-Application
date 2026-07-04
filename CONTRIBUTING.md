# Contributing / working on this project

This is an education & portfolio project. Contributions and experiments are
welcome; the notes below keep things reproducible.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt -r requirements-dev.txt
```

## Before you push

Run the tests — CI runs the same ones on every push:

```bash
python tests/run_offline_checks.py   # zero-dependency smoke check
pytest -m "not slow"                 # fast unit tests
pytest                               # everything, incl. end-to-end on synthetic data
```

All tests use a tiny **synthetic** dataset, so you do not need APTOS to run them.

## Making changes

- **Tunable settings go in `config.yaml`**, not hard-coded in scripts. If you add
  a setting, read it from config and document it with a `# why` comment.
- Keep `src/` modules importable without a GPU. Heavy imports (torch, sklearn,
  albumentations) stay lazy where they already are, so the light tests keep working.
- If you change the model architecture, update the parameter-count assertion in
  `tests/test_models.py` (and the number quoted in the README).
- Never commit data or model weights (`.gitignore` blocks `data/` and `*.pt`).

## Honesty rule

This is **not a medical device**. Don't add language to the app, docs, or README
that implies clinical validity. Keep the disclaimers intact.
