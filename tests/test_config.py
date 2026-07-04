"""Validate config.yaml itself — cheap guardrails that catch typos/edits."""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"


def load():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def test_config_file_exists():
    assert CONFIG.exists(), "config.yaml missing at repo root"


def test_required_top_level_sections():
    cfg = load()
    for key in ["project", "paths", "classes", "image", "preprocess",
                "split", "train", "imbalance", "model"]:
        assert key in cfg, f"config.yaml missing section '{key}'"


def test_classes_consistent():
    cfg = load()
    assert cfg["classes"]["num_classes"] == 5
    assert len(cfg["classes"]["names"]) == 5
    assert 0 <= cfg["classes"]["referable_threshold"] <= 4


def test_image_normalization_stats_are_triples():
    cfg = load()
    assert len(cfg["image"]["mean"]) == 3
    assert len(cfg["image"]["std"]) == 3
    assert cfg["image"]["size"] > 0


def test_model_name_is_known():
    cfg = load()
    valid = {"custom_cnn", "resnet50", "efficientnet_b0", "efficientnet_b3", "densenet121"}
    assert cfg["model"]["name"] in valid


def test_split_fractions_sane():
    cfg = load()
    assert 0 < cfg["split"]["val_size"] < 1
    assert 0 < cfg["split"]["test_size"] < 1
    assert cfg["split"]["val_size"] + cfg["split"]["test_size"] < 1
