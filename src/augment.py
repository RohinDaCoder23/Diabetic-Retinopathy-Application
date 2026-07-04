"""
augment.py — RANDOM data augmentation + normalization (Albumentations).

Two transform pipelines:
    build_train_transforms(cfg) : random perturbations + normalize + to-tensor.
        Runs ONLY on training images, re-rolled every epoch, so the model sees
        more variety and generalizes better.
    build_eval_transforms(cfg)  : normalize + to-tensor only. Deterministic, so
        validation/test scores are stable and comparable.

WHY normalization lives here (not in preprocess.py):
    Both pipelines end with the SAME Normalize(ImageNet mean/std) + ToTensorV2,
    so there is exactly one definition shared by train and eval. Pretrained
    backbones were trained on ImageNet-normalized inputs, so we must match it.

WHY the augmentations are GENTLE:
    DR grade is determined by TINY lesions — microaneurysms a few pixels wide.
    Aggressive crops, heavy zoom, or strong elastic warps can erase or invent
    that evidence, teaching the model the wrong thing. Fundus images also have
    no canonical orientation, so flips and rotations are safe and realistic;
    mild brightness/contrast jitter mimics different cameras. We deliberately
    avoid anything that could destroy lesion-level detail.
"""

from __future__ import annotations


def build_train_transforms(cfg: dict):
    """Training augmentation pipeline (Albumentations Compose)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    mean = cfg["image"]["mean"]
    std = cfg["image"]["std"]

    # Note: images arriving here are already FOV-cropped, Ben-Graham'd, and
    # resized by preprocess.build_preprocess(), so we do NOT resize again.
    return A.Compose([
        A.HorizontalFlip(p=0.5),          # retina has no left/right canonical side
        A.VerticalFlip(p=0.5),            # ...nor top/bottom
        A.Rotate(limit=20, p=0.6, border_mode=0),  # small rotations; black fill
        A.RandomBrightnessContrast(       # mimic camera/exposure variation
            brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.ShiftScaleRotate(               # MILD shift/zoom only — keep lesions in frame
            shift_limit=0.05, scale_limit=0.10, rotate_limit=0,
            border_mode=0, p=0.3),
        A.Normalize(mean=mean, std=std),  # ImageNet stats for pretrained nets
        ToTensorV2(),                     # HWC uint8/float -> CHW float tensor
    ])


def build_eval_transforms(cfg: dict):
    """Validation/test pipeline: normalize + to-tensor only (no randomness)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    mean = cfg["image"]["mean"]
    std = cfg["image"]["std"]
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def denormalize(tensor, cfg: dict):
    """Undo Normalize for visualization (e.g. showing the image behind Grad-CAM).

    Takes a CHW tensor that was ImageNet-normalized and returns an HWC float
    array in [0, 1] suitable for matplotlib.
    """
    import numpy as np
    import torch

    mean = torch.tensor(cfg["image"]["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["image"]["std"]).view(3, 1, 1)
    img = tensor.detach().cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return np.ascontiguousarray(img)
