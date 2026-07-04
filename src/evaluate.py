"""
evaluate.py — held-out test-set evaluation + figures.

Run from the repo root:

    python src/evaluate.py --config config.yaml --model resnet50

Loads the best checkpoint for a model, runs it on the TEST split (touched only
here, once), and reports the full metric suite:

    * accuracy (overall) — included, but MISLEADING under imbalance (a model
      that always predicts "No DR" scores ~50% while missing every patient).
    * per-class + macro precision / recall / F1 — recall on severe grades is
      what actually matters clinically (catching sick patients).
    * ROC-AUC (one-vs-rest, macro) — threshold-independent separability.
    * confusion matrix — see exactly which grades get confused.
    * Quadratic Weighted Kappa (QWK) — THE standard DR-grading metric.

WHY QWK is the headline metric:
    The grades are ordinal (0..4) and the dataset is imbalanced. QWK measures
    agreement with the true grade while penalizing errors by how FAR off they
    are (squared distance) — so predicting 4 when the truth is 0 is punished far
    more than predicting 1. It also corrects for agreement expected by chance.
    This matches how clinicians think about severity and is the metric the
    original Kaggle competition used.

Outputs:
    reports/metrics_<model>.json   (all numbers)
    reports/figures/confusion_matrix_<model>.png
    reports/figures/roc_curves_<model>.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .augment import build_eval_transforms
from .data import build_dataloaders
from .models import build_model
from .preprocess import build_preprocess
from .utils import get_device, load_checkpoint, load_config, plot_confusion_matrix, set_seed


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run the model over a loader, returning (y_true, y_pred, y_prob, ids)."""
    model.eval()
    ys, ps, probs, ids = [], [], [], []
    for batch in loader:
        images, labels = batch[0].to(device), batch[1]
        # test loader returns ids too (return_id=True in build_dataloaders)
        batch_ids = batch[2] if len(batch) > 2 else [None] * len(labels)
        logits = model(images)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        ys.extend(labels.numpy().tolist())
        ps.extend(prob.argmax(1).tolist())
        probs.extend(prob.tolist())
        ids.extend(list(batch_ids))
    return np.array(ys), np.array(ps), np.array(probs), ids


def compute_metrics(y_true, y_pred, y_prob, num_classes, class_names):
    """Compute the full metric dictionary."""
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["qwk"] = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    metrics["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class precision/recall/F1
    p = precision_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    r = recall_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    f = f1_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    metrics["per_class"] = {
        class_names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
        for i in range(num_classes)
    }

    # Macro one-vs-rest ROC-AUC (guard against classes missing in y_true)
    try:
        present = sorted(set(y_true.tolist()))
        if len(present) == num_classes:
            metrics["macro_roc_auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        else:
            metrics["macro_roc_auc"] = None  # some class absent in test split
    except Exception:
        metrics["macro_roc_auc"] = None

    # Referable DR (grade >= 2) binary view
    thr = 2
    yt_bin = (y_true >= thr).astype(int)
    yp_bin = (y_pred >= thr).astype(int)
    metrics["referable"] = {
        "precision": float(precision_score(yt_bin, yp_bin, zero_division=0)),
        "recall": float(recall_score(yt_bin, yp_bin, zero_division=0)),
        "f1": float(f1_score(yt_bin, yp_bin, zero_division=0)),
    }

    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=range(num_classes)
    ).tolist()
    return metrics


def plot_roc_curves(y_true, y_prob, num_classes, class_names, out_path):
    """One-vs-rest ROC curve per class."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(num_classes):
        if y_bin[:, i].sum() == 0:
            continue  # class absent in test split
        fpr, tpr, _ = roc_curve(y_bin[:, i], np.array(y_prob)[:, i])
        ax.plot(fpr, tpr, label=class_names[i])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (one-vs-rest)"); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as _plt; _plt.close(fig)


def evaluate(cfg, model_name=None):
    if model_name:
        cfg["model"]["name"] = model_name
    name = cfg["model"]["name"]
    set_seed(cfg["project"]["seed"])
    device = get_device()
    num_classes = cfg["classes"]["num_classes"]
    class_names = cfg["classes"]["names"]

    # Build test loader (same preprocessing/eval transforms as training).
    loaders = build_dataloaders(
        cfg, train_transform=None,
        eval_transform=build_eval_transforms(cfg),
        preprocess=build_preprocess(cfg),
    )
    test_loader = loaders["test"]

    # Load best checkpoint.
    ckpt_path = Path(cfg["paths"]["models_dir"]) / f"{name}_best.pt"
    ckpt = load_checkpoint(ckpt_path, map_location=device.type)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    y_true, y_pred, y_prob, _ = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes, class_names)
    metrics["model"] = name
    metrics["n_test"] = int(len(y_true))

    reports_dir = Path(cfg["paths"]["reports_dir"]); reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / f"metrics_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fig_dir = Path(cfg["paths"]["figures_dir"])
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]), class_names,
        fig_dir / f"confusion_matrix_{name}.png", title=f"Confusion matrix — {name}",
    )
    plot_roc_curves(y_true, y_prob, num_classes, class_names, fig_dir / f"roc_curves_{name}.png")

    print(f"[evaluate] {name}: accuracy={metrics['accuracy']:.3f} | "
          f"QWK={metrics['qwk']:.3f} | macroF1={metrics['macro_f1']:.3f} | "
          f"referable recall={metrics['referable']['recall']:.3f}")
    print(f"[evaluate] saved reports/metrics_{name}.json + figures")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained DR classifier.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), model_name=args.model)


if __name__ == "__main__":
    main()
