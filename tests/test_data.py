"""Data pipeline tests: synthetic generation, labels, splits, imbalance helpers.

The label/generation checks are torch-free. Split logic needs scikit-learn and
the imbalance helpers need torch, so those importorskip.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data


def test_generate_synthetic_dataset(tmp_path):
    root = tmp_path / "aptos2019"
    data.generate_synthetic_dataset(root, n_per_class=[6, 3, 4, 2, 5], image_size=64, seed=0)
    assert (root / "train.csv").exists()
    imgs = list((root / "train_images").glob("*.png"))
    assert len(imgs) == 6 + 3 + 4 + 2 + 5
    df = pd.read_csv(root / "train.csv")
    assert set(df.columns) == {"id_code", "diagnosis"}
    assert sorted(df["diagnosis"].unique().tolist()) == [0, 1, 2, 3, 4]


def test_read_labels_attaches_paths(synthetic_data):
    cfg = {"paths": {"train_csv": str(synthetic_data / "train.csv"),
                     "train_images": str(synthetic_data / "train_images")},
           "classes": {"num_classes": 5}}
    df = data.read_labels(cfg)
    assert {"id_code", "diagnosis", "path"} <= set(df.columns)
    assert all(Path(p).exists() for p in df["path"])


def test_read_labels_missing_csv_raises(tmp_path):
    cfg = {"paths": {"train_csv": str(tmp_path / "nope.csv"),
                     "train_images": str(tmp_path / "imgs")},
           "classes": {"num_classes": 5}}
    with pytest.raises(FileNotFoundError):
        data.read_labels(cfg)


def test_class_distribution_counts_all_grades(synthetic_data):
    cfg = {"paths": {"train_csv": str(synthetic_data / "train.csv"),
                     "train_images": str(synthetic_data / "train_images")},
           "classes": {"num_classes": 5}}
    df = data.read_labels(cfg)
    dist = data.class_distribution(df, 5)
    assert list(dist.index) == [0, 1, 2, 3, 4]
    assert dist.sum() == len(df)


@pytest.mark.needs_sklearn
def test_make_splits_are_disjoint_and_stratified(cfg):
    pytest.importorskip("sklearn")
    df = data.read_labels(cfg)
    train_df, val_df, test_df = data.make_splits(df, cfg)
    n = len(df)
    # Partition: sizes add up, no overlap in ids.
    assert len(train_df) + len(val_df) + len(test_df) == n
    ids = set(train_df.id_code) | set(val_df.id_code) | set(test_df.id_code)
    assert len(ids) == n
    # Every split contains every class (stratification worked).
    for part in (train_df, val_df, test_df):
        assert set(part["diagnosis"].unique()) == {0, 1, 2, 3, 4}


@pytest.mark.needs_sklearn
def test_make_splits_reproducible(cfg):
    pytest.importorskip("sklearn")
    df = data.read_labels(cfg)
    a = data.make_splits(df, cfg)[0]
    b = data.make_splits(df, cfg)[0]
    assert a["id_code"].tolist() == b["id_code"].tolist()


@pytest.mark.needs_torch
def test_class_weights_upweight_rare_classes(synthetic_data):
    pytest.importorskip("torch")
    cfg = {"paths": {"train_csv": str(synthetic_data / "train.csv"),
                     "train_images": str(synthetic_data / "train_images")},
           "classes": {"num_classes": 5}}
    df = data.read_labels(cfg)
    w = data.compute_class_weights(df, 5)
    assert tuple(w.shape) == (5,)
    # Weights average ~1.0 by construction.
    assert abs(float(w.mean()) - 1.0) < 1e-4
    # Rarer class (3, count 10) should weigh more than the most common (0, count 20).
    dist = data.class_distribution(df, 5)
    rare = int(dist.idxmin())
    common = int(dist.idxmax())
    assert float(w[rare]) > float(w[common])


@pytest.mark.needs_torch
def test_weighted_sampler_length_matches(synthetic_data):
    pytest.importorskip("torch")
    cfg = {"paths": {"train_csv": str(synthetic_data / "train.csv"),
                     "train_images": str(synthetic_data / "train_images")},
           "classes": {"num_classes": 5}}
    df = data.read_labels(cfg)
    sampler = data.make_weighted_sampler(df, 5)
    assert len(list(sampler)) == len(df)


@pytest.mark.needs_torch
@pytest.mark.needs_albumentations
def test_build_dataloaders_yields_correct_tensor_shapes(cfg):
    pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from src.augment import build_eval_transforms, build_train_transforms
    from src.preprocess import build_preprocess

    loaders = data.build_dataloaders(
        cfg,
        train_transform=build_train_transforms(cfg),
        eval_transform=build_eval_transforms(cfg),
        preprocess=build_preprocess(cfg),
    )
    for key in ("train", "val", "test", "class_weights", "train_df", "val_df", "test_df"):
        assert key in loaders
    images, labels = next(iter(loaders["train"]))
    size = cfg["image"]["size"]
    assert images.shape[1:] == (3, size, size)
    assert images.shape[0] == cfg["train"]["batch_size"]
    assert labels.min() >= 0 and labels.max() <= 4
