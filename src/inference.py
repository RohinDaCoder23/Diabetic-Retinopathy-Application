"""
inference.py — single-image inference helpers shared by the Streamlit app.

Keeping this separate from the app means the prediction logic is testable and
reused exactly as in training (same preprocessing + normalization), so the app
can never silently diverge from how the model was trained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def list_available_checkpoints(cfg: dict) -> dict:
    """Return {model_name: path} for checkpoints that exist on disk."""
    models_dir = Path(cfg["paths"]["models_dir"])
    found = {}
    from .models import VALID_MODELS

    for name in VALID_MODELS:
        p = models_dir / f"{name}_best.pt"
        if p.exists():
            found[name] = str(p)
    return found


def load_model(cfg: dict, model_name: str, device=None):
    """Build the model and load its best checkpoint. Returns (model, device)."""
    import torch

    from .models import build_model
    from .utils import get_device, load_checkpoint

    device = device or get_device(verbose=False)
    cfg = {**cfg, "model": {**cfg["model"], "name": model_name}}
    ckpt_path = Path(cfg["paths"]["models_dir"]) / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No trained weights for '{model_name}' at {ckpt_path}. "
            f"Train it first (see docs/colab_quickstart.md)."
        )
    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(ckpt_path, map_location=device.type)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def preprocess_image(rgb_uint8: np.ndarray, cfg: dict):
    """Apply the SAME preprocessing + eval transform used in training.

    Returns a CHW float tensor ready for the model, plus the preprocessed
    uint8 RGB image (for display / Grad-CAM background).
    """
    from .augment import build_eval_transforms
    from .preprocess import build_preprocess

    pre = build_preprocess(cfg)
    clean = pre(rgb_uint8)  # FOV crop -> Ben-Graham -> (CLAHE) -> resize
    tensor = build_eval_transforms(cfg)(image=clean)["image"]  # normalize + to-tensor
    return tensor, clean


def predict(model, tensor, device) -> Tuple[int, np.ndarray]:
    """Return (predicted_grade, probability_vector) for one CHW tensor."""
    import torch

    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return int(probs.argmax()), probs


def explain(model, tensor, model_name: str, target_class: int, cfg: dict):
    """Return (rgb_float01, gradcam_overlay_uint8) for the prediction."""
    from .gradcam import gradcam_single
    from .models import get_gradcam_target_layer

    target_layer = get_gradcam_target_layer(model, model_name)
    return gradcam_single(model, tensor, target_layer, target_class, cfg)
