"""Preprocessing tests — pure OpenCV/numpy, run in any environment."""
from __future__ import annotations

import numpy as np
import pytest

from src import preprocess as pp


def _fake_fundus(h=300, w=400, cx=220, cy=150, r=120):
    """Black canvas with an off-center bright disc (a stand-in retina)."""
    import cv2

    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (cx, cy), r, (30, 40, 150), -1)
    return img


def test_crop_field_of_view_trims_black_border():
    img = _fake_fundus()
    cropped = pp.crop_field_of_view(img)
    # Should be smaller than the original canvas but non-empty.
    assert cropped.shape[0] < img.shape[0]
    assert cropped.shape[1] < img.shape[1]
    assert cropped.size > 0


def test_crop_all_black_is_safe():
    black = np.zeros((50, 50, 3), np.uint8)
    out = pp.crop_field_of_view(black)
    assert out.shape == black.shape  # returns unchanged, no crash


def test_ben_graham_returns_uint8_same_shape():
    img = _fake_fundus()
    out = pp.ben_graham(img, sigma=10)
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert out.min() >= 0 and out.max() <= 255


def test_apply_clahe_preserves_shape_and_channels():
    img = _fake_fundus()
    out = pp.apply_clahe(img)
    assert out.shape == img.shape


def test_resize_to_square():
    img = _fake_fundus()
    out = pp.resize(img, 224)
    assert out.shape == (224, 224, 3)


def test_build_preprocess_is_deterministic():
    cfg = {"image": {"size": 128},
           "preprocess": {"crop_fov": True, "ben_graham": True, "clahe": False,
                          "ben_graham_sigma": 10}}
    fn = pp.build_preprocess(cfg)
    img = _fake_fundus()
    a = fn(img.copy())
    b = fn(img.copy())
    assert a.shape == (128, 128, 3)
    # Deterministic: identical input -> identical output (this is the whole point
    # of preprocessing vs augmentation).
    assert np.array_equal(a, b)


def test_step_by_step_returns_all_stages():
    cfg = {"image": {"size": 128}, "preprocess": {"ben_graham_sigma": 10}}
    stages = pp.step_by_step(_fake_fundus(), cfg)
    assert set(stages) == {"original", "crop_fov", "ben_graham", "clahe", "resize"}
    assert stages["resize"].shape == (128, 128, 3)


def test_clahe_toggle_changes_output():
    img = _fake_fundus()
    base = {"image": {"size": 128},
            "preprocess": {"crop_fov": True, "ben_graham": True, "ben_graham_sigma": 10}}
    off = pp.build_preprocess({**base, "preprocess": {**base["preprocess"], "clahe": False}})(img.copy())
    on = pp.build_preprocess({**base, "preprocess": {**base["preprocess"], "clahe": True}})(img.copy())
    assert off.shape == on.shape
