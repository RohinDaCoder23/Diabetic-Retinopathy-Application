"""
Shared pytest fixtures for the diabetic-retinopathy test suite.

Design goals:
  * No real dataset required — a tiny SYNTHETIC APTOS-like dataset is generated
    on the fly (deterministic, seed-fixed).
  * No network required — transfer models are built with pretrained=False, so
    ImageNet weights are never downloaded during tests.
  * Heavy deps (torch, sklearn, albumentations) are imported lazily inside the
    tests that need them via `pytest.importorskip`, so the light tests still run
    in a minimal environment.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

# Make the repo root importable (so `import src...` works) even if pytest.ini
# pythonpath is not honored by an older pytest.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Torch-free import: src.data only needs numpy + pandas at module load time.
from src import data as data_mod  # noqa: E402


CLASS_NAMES = [
    "0 - No DR",
    "1 - Mild",
    "2 - Moderate",
    "3 - Severe",
    "4 - Proliferative DR",
]

# Enough samples per class that a stratified 3-way split always has >=1 per
# class in every split. Kept small so tests are fast.
N_PER_CLASS = [20, 10, 15, 10, 12]


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory) -> Path:
    """Generate a tiny fake APTOS dataset once per test session.

    Returns the dataset root containing train.csv and train_images/.
    """
    root = tmp_path_factory.mktemp("aptos_synth") / "aptos2019"
    data_mod.generate_synthetic_dataset(
        root, n_per_class=N_PER_CLASS, image_size=96, seed=123
    )
    return root


def _deep_update(base: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def build_config(data_root: Path, work_dir: Path, **overrides) -> dict:
    """Return a complete config dict pointing at the synthetic dataset.

    Small image size and batch size keep model tests fast on CPU. Callers can
    override any nested key, e.g. build_config(..., model={"name": "resnet50"}).
    """
    cfg = {
        "project": {"name": "dr-test", "seed": 42},
        "paths": {
            "data_dir": str(data_root),
            "train_csv": str(data_root / "train.csv"),
            "train_images": str(data_root / "train_images"),
            "test_csv": str(data_root / "test.csv"),
            "test_images": str(data_root / "test_images"),
            "models_dir": str(work_dir / "models"),
            "reports_dir": str(work_dir / "reports"),
            "figures_dir": str(work_dir / "reports" / "figures"),
        },
        "classes": {
            "num_classes": 5,
            "names": list(CLASS_NAMES),
            "referable_threshold": 2,
        },
        "image": {
            "size": 64,
            "channels": 3,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "preprocess": {
            "crop_fov": True,
            "ben_graham": True,
            "ben_graham_sigma": 10,
            "clahe": False,
            "clahe_clip": 2.0,
            "clahe_grid": 8,
        },
        "split": {"val_size": 0.2, "test_size": 0.2, "stratify": True},
        "train": {
            "batch_size": 4,
            "epochs": 1,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "scheduler": "none",
            "warmup_epochs": 0,
            "early_stop_patience": 3,
            "early_stop_metric": "qwk",
            "amp": False,          # AMP is auto-disabled on CPU anyway
            "num_workers": 0,      # single-process = deterministic + no fork issues
            "grad_clip": 5.0,
        },
        "imbalance": {"strategy": "weighted_loss", "use_sampler": False, "focal_gamma": 2.0},
        "model": {
            "name": "custom_cnn",
            "pretrained": False,   # never download ImageNet weights in tests
            "dropout": 0.3,
            "freeze_backbone": False,
        },
    }
    return _deep_update(copy.deepcopy(cfg), overrides)


@pytest.fixture
def cfg(synthetic_data, tmp_path):
    """A per-test config dict with isolated models/ and reports/ dirs."""
    return build_config(synthetic_data, tmp_path)


@pytest.fixture
def cfg_factory(synthetic_data, tmp_path):
    """Return a builder so a test can request a config with overrides."""
    def _make(**overrides):
        return build_config(synthetic_data, tmp_path, **overrides)
    return _make
