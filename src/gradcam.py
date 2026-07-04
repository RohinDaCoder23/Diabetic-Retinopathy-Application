"""
gradcam.py — Grad-CAM explainability heatmaps.

Run from the repo root:

    python src/gradcam.py --config config.yaml --model resnet50

Grad-CAM ("Gradient-weighted Class Activation Mapping") shows WHERE the model
looked. Intuition: take the last convolutional feature maps (which still have
spatial layout), weight each map by how much it pushes up the score for the
predicted grade (that "how much" is the gradient), sum them, and you get a
coarse heat map of the regions that drove the decision. Hot = influential.

How it works, step by step:
    1. Forward the image; pick the target class (the predicted grade).
    2. Backprop that class score to the last conv layer's feature maps.
    3. Global-average-pool the gradients -> one weight per feature map.
    4. Weighted sum of feature maps -> ReLU (keep positive evidence) -> upscale
       to the image size -> overlay as a heatmap.

How to READ it (and a caution):
    A TRUSTWORTHY DR heatmap concentrates on lesions — microaneurysms,
    hemorrhages, exudates — not on the image border, the optic disc by default,
    or black background. Grad-CAM is a *coarse* localizer and only shows
    correlation with the output, not true clinical reasoning. A confident,
    well-placed heatmap can still sit on a WRONG prediction. Use it as a sanity
    check and a teaching aid, NOT as proof the model is right.

This module uses the `grad-cam` package (pytorch-grad-cam) from requirements.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .augment import build_eval_transforms, denormalize
from .data import build_dataloaders
from .models import build_model, get_gradcam_target_layer
from .preprocess import build_preprocess
from .utils import get_device, load_checkpoint, load_config, set_seed


def make_overlay(rgb_float01, grayscale_cam):
    """Blend a [0,1] RGB image with a Grad-CAM heatmap -> uint8 RGB."""
    from pytorch_grad_cam.utils.image import show_cam_on_image

    return show_cam_on_image(rgb_float01, grayscale_cam, use_rgb=True)


def generate_gallery(cfg, model_name=None, per_grade=2, max_errors=4):
    """Save (original | Grad-CAM overlay) panels per grade, plus some errors.

    For each true grade we save a few correctly-predicted examples, and we also
    collect a handful of mistakes (predicted != true) — errors are often the most
    instructive heatmaps to discuss in a presentation.
    """
    from pytorch_grad_cam import GradCAM

    if model_name:
        cfg["model"]["name"] = model_name
    name = cfg["model"]["name"]
    set_seed(cfg["project"]["seed"])
    device = get_device()
    class_names = cfg["classes"]["names"]

    loaders = build_dataloaders(
        cfg, train_transform=None,
        eval_transform=build_eval_transforms(cfg),
        preprocess=build_preprocess(cfg),
    )
    test_loader = loaders["test"]

    ckpt = load_checkpoint(Path(cfg["paths"]["models_dir"]) / f"{name}_best.pt",
                           map_location=device.type)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    target_layer = get_gradcam_target_layer(model, name)
    cam = GradCAM(model=model, target_layers=[target_layer])

    out_dir = Path(cfg["paths"]["figures_dir"]) / f"gradcam_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    saved_per_grade = {g: 0 for g in range(cfg["classes"]["num_classes"])}
    errors_saved = 0

    for batch in test_loader:
        images, labels = batch[0].to(device), batch[1]
        ids = batch[2] if len(batch) > 2 else [f"img{i}" for i in range(len(labels))]
        with torch.no_grad():
            preds = model(images).argmax(1).cpu().numpy()

        for i in range(images.size(0)):
            true_g, pred_g = int(labels[i]), int(preds[i])
            is_error = true_g != pred_g
            # keep a few correct per grade + a few errors
            if not is_error and saved_per_grade[true_g] >= per_grade:
                continue
            if is_error and errors_saved >= max_errors:
                if saved_per_grade[true_g] >= per_grade:
                    continue

            grayscale = cam(input_tensor=images[i:i+1],
                            targets=[ClassifierOutputTarget(pred_g)])[0]
            rgb = denormalize(images[i], cfg)
            overlay = make_overlay(rgb, grayscale)
            _save_pair(rgb, overlay, true_g, pred_g, class_names,
                       out_dir / f"{ids[i]}_t{true_g}_p{pred_g}.png")

            if is_error:
                errors_saved += 1
            else:
                saved_per_grade[true_g] += 1

        if all(v >= per_grade for v in saved_per_grade.values()) and errors_saved >= max_errors:
            break

    print(f"[gradcam] saved gallery to {out_dir}")
    return out_dir


def _save_pair(rgb01, overlay, true_g, pred_g, class_names, out_path):
    """Save a side-by-side original | heatmap figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    axes[0].imshow(rgb01); axes[0].set_title("original", fontsize=10); axes[0].axis("off")
    axes[1].imshow(overlay); axes[1].axis("off")
    correct = "correct" if true_g == pred_g else "ERROR"
    axes[1].set_title(f"Grad-CAM ({correct})", fontsize=10)
    fig.suptitle(f"true: {class_names[true_g]}   |   pred: {class_names[pred_g]}", fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def gradcam_single(model, image_tensor, target_layer, target_class, cfg):
    """Convenience: Grad-CAM for ONE preprocessed tensor (used by the app).

    Returns (rgb_float01, overlay_uint8). image_tensor is CHW, already
    normalized. target_class is the grade to explain (usually the prediction).
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    device = next(model.parameters()).device
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale = cam(input_tensor=image_tensor.unsqueeze(0).to(device),
                    targets=[ClassifierOutputTarget(int(target_class))])[0]
    rgb = denormalize(image_tensor, cfg)
    return rgb, make_overlay(rgb, grayscale)


def main():
    ap = argparse.ArgumentParser(description="Generate Grad-CAM gallery.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--per-grade", type=int, default=2)
    args = ap.parse_args()
    generate_gallery(load_config(args.config), model_name=args.model, per_grade=args.per_grade)


if __name__ == "__main__":
    main()
