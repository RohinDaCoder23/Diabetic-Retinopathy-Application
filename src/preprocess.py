"""
preprocess.py — DETERMINISTIC image cleanup applied to every image.

Preprocessing vs augmentation (the key distinction):
    * Preprocessing is DETERMINISTIC. The same input always produces the same
      output. It runs at train, validation, test, AND inference time, so the
      model always sees images in the same standardized form.
    * Augmentation (see augment.py) is RANDOM and runs only on training data.

Pipeline (each step is toggled in config.yaml -> preprocess):
    1. crop_field_of_view : trim black borders, crop the circular retina.
    2. ben_graham         : subtract a blurred local average (color/lighting norm).
    3. clahe              : optional local contrast enhancement.
    4. resize             : to the model input size (e.g. 224x224).
       (Normalization with ImageNet mean/std happens in augment.py, so train and
        eval transforms share exactly one definition of it.)

Every function takes and returns an HxWx3 uint8 RGB numpy array unless noted,
so steps compose cleanly and are easy to visualize before/after.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Field-of-view crop
# ---------------------------------------------------------------------------
def crop_field_of_view(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Trim black borders so the image is mostly retina, not background.

    WHY: fundus photos sit on a black canvas with lots of empty space and the
    retina is an off-center circle. Cropping to the informative region means the
    model spends its capacity (and our limited resolution) on actual retina, and
    makes images from different cameras more consistent.

    Method: build a mask of "non-dark" pixels (grayscale > tol) and crop to its
    bounding box. Robust to small dark specks via the tolerance.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    mask = gray > tol
    if not mask.any():
        return img  # all-black safety net: return unchanged

    coords = np.ix_(mask.any(axis=1), mask.any(axis=0))
    if img.ndim == 3:
        cropped = img[coords[0].ravel()][:, coords[1].ravel()]
    else:
        cropped = img[coords]
    return cropped


def circle_crop(img: np.ndarray) -> np.ndarray:
    """Crop to the largest centered square then mask to a circle.

    Optional stronger version of FOV cropping: after trimming borders, keep only
    a circular region (the retina is circular), zeroing the corners. Helps remove
    camera-edge artifacts. Used when you want maximum standardization.
    """
    img = crop_field_of_view(img)
    h, w = img.shape[:2]
    s = min(h, w)
    # center crop to square
    top = (h - s) // 2
    left = (w - s) // 2
    img = img[top:top + s, left:left + s]
    # circular mask
    mask = np.zeros((s, s), dtype=np.uint8)
    cv2.circle(mask, (s // 2, s // 2), s // 2, 1, -1)
    return img * mask[..., None]


# ---------------------------------------------------------------------------
# 2. Ben-Graham color normalization
# ---------------------------------------------------------------------------
def ben_graham(img: np.ndarray, sigma: int = 10) -> np.ndarray:
    """Ben-Graham local-average subtraction (the classic Kaggle DR trick).

    WHY: fundus cameras differ wildly in color and lighting. Ben Graham (winner
    of the 2015 Kaggle DR competition) popularized subtracting a heavily blurred
    version of the image, which removes slow lighting gradients and dramatically
    boosts the visibility of small lesions (microaneurysms, hemorrhages).

    Formula: out = 4*img - 4*GaussianBlur(img) + 128
    The +128 re-centers the result into a visible range. The factor 4 amplifies
    the high-frequency detail (the lesions) that survived the subtraction.
    """
    img = img.astype(np.float32)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    out = cv2.addWeighted(img, 4, blur, -4, 128)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. CLAHE
# ---------------------------------------------------------------------------
def apply_clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on the luminance channel.

    WHY: CLAHE boosts *local* contrast, which can make faint lesions pop. We
    apply it on the L channel in LAB space so colors aren't distorted. The clip
    limit prevents over-amplifying noise.

    NOTE: CLAHE and Ben-Graham both enhance contrast; using both can be
    redundant or harsh, so CLAHE is OFF by default in config.yaml. Try one.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------
# 4. Resize
# ---------------------------------------------------------------------------
def resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to a square (size x size) with area interpolation (good for shrink)."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Composed preprocessing function (config-driven)
# ---------------------------------------------------------------------------
def build_preprocess(cfg: dict):
    """Return a single callable preprocess(img_uint8_rgb) -> img_uint8_rgb.

    Reads the `preprocess` and `image` sections of config.yaml and composes the
    enabled steps in the right order. Normalization is intentionally NOT here —
    it lives in augment.py so train/eval transforms share one definition.

    The returned function is what gets passed to APTOSDataset(preprocess=...).
    """
    p = cfg.get("preprocess", {})
    size = cfg["image"]["size"]

    def _fn(img: np.ndarray) -> np.ndarray:
        if p.get("crop_fov", True):
            img = crop_field_of_view(img)
        if p.get("ben_graham", True):
            img = ben_graham(img, sigma=p.get("ben_graham_sigma", 10))
        if p.get("clahe", False):
            img = apply_clahe(img, clip=p.get("clahe_clip", 2.0), grid=p.get("clahe_grid", 8))
        img = resize(img, size)
        return img

    return _fn


# Names of steps for the before/after visualization in notebook 02.
PREPROCESS_STEPS = ["original", "crop_fov", "ben_graham", "clahe", "resize"]


def step_by_step(img: np.ndarray, cfg: dict) -> dict:
    """Return a dict of intermediate images for the 02_preprocessing notebook.

    Always computes every step (regardless of config toggles) so you can SEE
    what each one does, before deciding which to enable in config.yaml.
    """
    size = cfg["image"]["size"]
    p = cfg.get("preprocess", {})
    out = {"original": img.copy()}
    a = crop_field_of_view(img)
    out["crop_fov"] = a
    b = ben_graham(a, sigma=p.get("ben_graham_sigma", 10))
    out["ben_graham"] = b
    c = apply_clahe(b, clip=p.get("clahe_clip", 2.0), grid=p.get("clahe_grid", 8))
    out["clahe"] = c
    out["resize"] = resize(b, size)  # show resize on the Ben-Graham result (default path)
    return out
