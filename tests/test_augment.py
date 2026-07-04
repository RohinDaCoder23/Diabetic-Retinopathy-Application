"""Augmentation/transform tests. Need albumentations (+ torch for tensors)."""
from __future__ import annotations

import numpy as np
import pytest


def _cfg():
    return {"image": {"size": 64, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}


@pytest.mark.needs_albumentations
def test_eval_transform_outputs_chw_float_tensor():
    torch = pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from src.augment import build_eval_transforms

    img = np.random.randint(0, 255, (64, 64, 3), np.uint8)
    out = build_eval_transforms(_cfg())(image=img)["image"]
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 64, 64)      # channels-first
    assert out.dtype == torch.float32


@pytest.mark.needs_albumentations
def test_eval_transform_is_deterministic():
    pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from src.augment import build_eval_transforms

    img = np.random.randint(0, 255, (64, 64, 3), np.uint8)
    t = build_eval_transforms(_cfg())
    a = t(image=img)["image"]
    b = t(image=img)["image"]
    assert (a == b).all()  # eval transform has no randomness


@pytest.mark.needs_albumentations
def test_train_transform_shape_and_randomness():
    torch = pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    import random
    from src.augment import build_train_transforms

    img = np.random.randint(0, 255, (64, 64, 3), np.uint8)
    t = build_train_transforms(_cfg())
    out = t(image=img)["image"]
    assert out.shape == (3, 64, 64)
    # Over several draws, at least one should differ (augmentation is random).
    outs = [t(image=img)["image"] for _ in range(8)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:])


@pytest.mark.needs_albumentations
def test_denormalize_roundtrips_into_unit_range():
    torch = pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from src.augment import build_eval_transforms, denormalize

    img = np.random.randint(0, 255, (64, 64, 3), np.uint8)
    tensor = build_eval_transforms(_cfg())(image=img)["image"]
    back = denormalize(tensor, _cfg())
    assert back.shape == (64, 64, 3)
    assert back.min() >= 0.0 and back.max() <= 1.0
