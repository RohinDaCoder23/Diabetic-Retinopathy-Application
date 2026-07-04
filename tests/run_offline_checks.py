#!/usr/bin/env python3
"""
run_offline_checks.py — a zero-extra-dependency smoke check.

WHY this exists in addition to the pytest suite:
    The full pytest suite needs torch + scikit-learn + albumentations. In a
    locked-down environment (no internet to `pip install`), you can still verify
    the entire image-processing core with only numpy + OpenCV + pandas + PyYAML,
    which ship almost everywhere. This script does exactly that, and it ALSO
    runs the heavier checks automatically IF torch/sklearn happen to be present.

Run it from the repo root:
    python tests/run_offline_checks.py

Exit code 0 = every check that could run passed. Skipped checks (missing optional
deps) do not fail the run; they are reported so you know what was not exercised.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results = []


def check(name):
    def deco(fn):
        try:
            fn()
            results.append((PASS, name, ""))
        except _Skip as s:
            results.append((SKIP, name, str(s)))
        except Exception as e:  # noqa: BLE001
            results.append((FAIL, name, f"{type(e).__name__}: {e}"))
    return deco


class _Skip(Exception):
    pass


def need(module):
    try:
        __import__(module)
    except Exception:
        raise _Skip(f"needs {module}")


# ---------------------------------------------------------------------------
# Torch-free checks (numpy / cv2 / pandas / yaml only)
# ---------------------------------------------------------------------------
@check("config.yaml is valid and complete")
def _():
    need("yaml")
    import yaml
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    for k in ["project", "paths", "classes", "image", "preprocess", "split",
              "train", "imbalance", "model"]:
        assert k in cfg, f"missing section {k}"
    assert cfg["classes"]["num_classes"] == 5
    assert len(cfg["classes"]["names"]) == 5


@check("preprocess pipeline runs on real pixels")
def _():
    need("cv2")
    import cv2, numpy as np
    from src import preprocess as pp
    img = np.zeros((300, 400, 3), np.uint8)
    cv2.circle(img, (220, 150), 120, (30, 40, 150), -1)
    assert pp.crop_field_of_view(img).shape[0] < img.shape[0]
    assert pp.ben_graham(img, 10).dtype == np.uint8
    out = pp.build_preprocess({"image": {"size": 224},
                               "preprocess": {"crop_fov": True, "ben_graham": True,
                                              "clahe": False, "ben_graham_sigma": 10}})(img)
    assert out.shape == (224, 224, 3)
    a = pp.resize(img, 128); b = pp.resize(img, 128)
    assert (a == b).all(), "preprocessing must be deterministic"


@check("synthetic dataset generates + labels read back")
def _():
    need("cv2")
    import pandas as pd
    from src import data
    tmp = Path(tempfile.mkdtemp()) / "aptos2019"
    data.generate_synthetic_dataset(tmp, n_per_class=[8, 3, 5, 2, 4], image_size=64, seed=1)
    n = len(list((tmp / "train_images").glob("*.png")))
    assert n == 22, n
    cfg = {"paths": {"train_csv": str(tmp / "train.csv"),
                     "train_images": str(tmp / "train_images")},
           "classes": {"num_classes": 5}}
    df = data.read_labels(cfg)
    assert {"id_code", "diagnosis", "path"} <= set(df.columns)
    assert data.class_distribution(df, 5).tolist() == [8, 3, 5, 2, 4]


@check("custom_cnn parameter count == 390,181 (analytic)")
def _():
    conv = lambda i, o, k=3: k * k * i * o
    bn = lambda o: 2 * o
    total = (conv(3, 32) + bn(32) + conv(32, 64) + bn(64) + conv(64, 128) + bn(128)
             + conv(128, 256) + bn(256) + (256 * 5 + 5))
    assert total == 390_181, total


# ---------------------------------------------------------------------------
# Heavier checks (auto-run only if torch is importable)
# ---------------------------------------------------------------------------
@check("stratified split partitions data (needs sklearn)")
def _():
    need("sklearn"); need("cv2")
    from src import data
    tmp = Path(tempfile.mkdtemp()) / "aptos2019"
    data.generate_synthetic_dataset(tmp, n_per_class=[20, 10, 15, 10, 12], image_size=48, seed=2)
    cfg = {"paths": {"train_csv": str(tmp / "train.csv"),
                     "train_images": str(tmp / "train_images")},
           "classes": {"num_classes": 5},
           "project": {"seed": 42},
           "split": {"val_size": 0.2, "test_size": 0.2, "stratify": True}}
    df = data.read_labels(cfg)
    tr, va, te = data.make_splits(df, cfg)
    assert len(tr) + len(va) + len(te) == len(df)
    for part in (tr, va, te):
        assert set(part["diagnosis"]) == {0, 1, 2, 3, 4}


@check("build + forward custom_cnn (needs torch)")
def _():
    need("torch")
    import torch
    from src.models import build_model
    m = build_model({"model": {"name": "custom_cnn", "pretrained": False, "dropout": 0.3},
                     "classes": {"num_classes": 5}}).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 5)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == 390_181, n


def main():
    print("\n  Offline checks for diabetic-retinopathy\n  " + "-" * 44)
    for status, name, detail in results:
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "—"}[status]
        line = f"  [{icon}] {status:4}  {name}"
        if detail:
            line += f"   ({detail})"
        print(line)
    n_pass = sum(r[0] == PASS for r in results)
    n_fail = sum(r[0] == FAIL for r in results)
    n_skip = sum(r[0] == SKIP for r in results)
    print("  " + "-" * 44)
    print(f"  {n_pass} passed, {n_skip} skipped, {n_fail} failed\n")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
