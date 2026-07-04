"""
End-to-end pipeline tests (marked `slow`): train -> checkpoint -> evaluate ->
inference -> grad-cam, all on the tiny synthetic dataset.

These need torch + sklearn + albumentations, so they importorskip and are also
tagged `slow` (deselect with `-m "not slow"`). They are the tests that prove the
FULL chain actually executes end to end — exactly what CI runs on every push.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _require_all():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    pytest.importorskip("albumentations")


def test_smoke_train_writes_checkpoint(cfg):
    _require_all()
    from src.train import train

    result = train(cfg, model_name="custom_cnn", smoke_test=True)
    ckpt = Path(result["checkpoint"])
    assert ckpt.exists(), "smoke training should write a checkpoint .pt"
    assert ckpt.suffix == ".pt"


def test_full_one_epoch_train_then_evaluate(cfg_factory):
    _require_all()
    import torch
    from src.train import train
    from src.evaluate import evaluate

    # 1 real epoch (not smoke) so evaluate has a genuine checkpoint to load.
    cfg = cfg_factory(model={"name": "custom_cnn", "pretrained": False,
                             "dropout": 0.3, "freeze_backbone": False})
    out = train(cfg, model_name="custom_cnn")
    assert Path(out["checkpoint"]).exists()

    metrics = evaluate(cfg, model_name="custom_cnn")
    # Metric suite is present and in-range.
    for key in ["accuracy", "qwk", "macro_f1", "macro_recall", "referable", "confusion_matrix"]:
        assert key in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert -1.0 <= metrics["qwk"] <= 1.0
    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (5, 5)
    assert cm.sum() == metrics["n_test"]

    # metrics json + figures were written
    reports = Path(cfg["paths"]["reports_dir"])
    assert (reports / "metrics_custom_cnn.json").exists()
    figs = Path(cfg["paths"]["figures_dir"])
    assert (figs / "confusion_matrix_custom_cnn.png").exists()


def test_inference_on_single_image(cfg):
    _require_all()
    import cv2
    from src.train import train
    from src.inference import load_model, preprocess_image, predict

    train(cfg, model_name="custom_cnn", smoke_test=True)

    model, device = load_model(cfg, "custom_cnn")
    # A synthetic image straight off disk.
    img_path = next(Path(cfg["paths"]["train_images"]).glob("*.png"))
    rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    tensor, clean = preprocess_image(rgb, cfg)
    grade, probs = predict(model, tensor, device)
    assert grade in range(5)
    assert len(probs) == 5
    assert abs(float(np.sum(probs)) - 1.0) < 1e-4  # softmax sums to 1
    assert clean.shape == (cfg["image"]["size"], cfg["image"]["size"], 3)


def test_list_available_checkpoints(cfg):
    _require_all()
    from src.train import train
    from src.inference import list_available_checkpoints

    assert list_available_checkpoints(cfg) == {}  # none yet
    train(cfg, model_name="custom_cnn", smoke_test=True)
    found = list_available_checkpoints(cfg)
    assert "custom_cnn" in found


def test_gradcam_single_image(cfg):
    _require_all()
    pytest.importorskip("pytorch_grad_cam")
    import cv2
    from src.train import train
    from src.inference import load_model, preprocess_image, predict, explain

    train(cfg, model_name="custom_cnn", smoke_test=True)
    model, device = load_model(cfg, "custom_cnn")
    img_path = next(Path(cfg["paths"]["train_images"]).glob("*.png"))
    rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    tensor, _ = preprocess_image(rgb, cfg)
    grade, _ = predict(model, tensor, device)
    rgb01, overlay = explain(model, tensor, "custom_cnn", grade, cfg)
    size = cfg["image"]["size"]
    assert rgb01.shape == (size, size, 3)
    assert overlay.shape == (size, size, 3)
