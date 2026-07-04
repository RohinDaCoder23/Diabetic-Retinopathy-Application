"""
train.py — the config-driven training engine.

Run from the repo root:

    # full training (model taken from config.yaml -> model.name)
    python src/train.py --config config.yaml

    # override the architecture on the command line
    python src/train.py --config config.yaml --model resnet50

    # quick sanity check: 1 epoch on a tiny subset (use before a long run)
    python src/train.py --config config.yaml --model resnet50 --smoke-test

What it does, in order:
    1. set seeds, pick device, load config.
    2. build DataLoaders (with preprocessing + augmentation + optional sampler).
    3. build the model from config.
    4. build loss (class-weighted CrossEntropy or focal), optimizer (AdamW),
       and LR scheduler (cosine / ReduceLROnPlateau / none).
    5. loop over epochs: train, validate, log metrics, step scheduler, early-stop.
    6. save the BEST checkpoint (by val metric) and training-curve plots.

Design choices (the "WHY", also in docs/pipeline.md):
    * Loss = class-weighted CrossEntropy. The dataset is imbalanced; weighting by
      inverse class frequency stops the model from ignoring rare, severe grades.
      Focal loss is offered as an alternative for very hard imbalance.
    * Optimizer = AdamW. Adam-style adaptive steps converge fast and need little
      tuning; the "W" (decoupled weight decay) regularizes more correctly.
    * Scheduler = cosine annealing by default. Smoothly lowers the LR so training
      settles into a good minimum; ReduceLROnPlateau is available as an
      alternative that drops the LR when val stops improving.
    * Early stopping on QWK. We stop when the validation metric hasn't improved
      for `early_stop_patience` epochs, saving compute and avoiding overfitting.
    * Mixed precision (AMP) when a GPU is present: faster and uses less memory,
      with no meaningful accuracy cost. Auto-disabled on CPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .augment import build_eval_transforms, build_train_transforms
from .data import build_dataloaders
from .models import build_model
from .preprocess import build_preprocess
from .utils import (
    get_device,
    get_logger,
    load_config,
    plot_training_curves,
    save_checkpoint,
    set_seed,
)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Multi-class focal loss: down-weights easy, well-classified examples.

    loss = -alpha_c * (1 - p_t)^gamma * log(p_t)
    With gamma=0 this reduces to (weighted) cross-entropy. Larger gamma focuses
    training harder on misclassified examples — useful under severe imbalance.
    """

    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)  # probability of the true class
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(cfg, class_weights, device):
    """Pick the loss from config.yaml -> imbalance.strategy."""
    strategy = cfg["imbalance"].get("strategy", "weighted_loss")
    w = class_weights.to(device) if class_weights is not None else None
    if strategy == "focal":
        return FocalLoss(gamma=cfg["imbalance"].get("focal_gamma", 2.0), weight=w)
    if strategy == "weighted_loss":
        return nn.CrossEntropyLoss(weight=w)
    return nn.CrossEntropyLoss()  # "none" or sampler-only


# ---------------------------------------------------------------------------
# Lightweight validation metrics (full suite lives in evaluate.py)
# ---------------------------------------------------------------------------
def quick_metrics(y_true, y_pred):
    """Return (quadratic weighted kappa, macro-F1) for early stopping."""
    from sklearn.metrics import cohen_kappa_score, f1_score

    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return float(qwk), float(macro_f1)


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------
def build_optimizer(cfg, model):
    params = [p for p in model.parameters() if p.requires_grad]
    lr = cfg["train"]["lr"]
    wd = cfg["train"].get("weight_decay", 1e-4)
    if cfg["train"].get("optimizer", "adamw").lower() == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def build_scheduler(cfg, optimizer):
    sched = cfg["train"].get("scheduler", "cosine")
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["train"]["epochs"]
        ), "epoch"
    if sched == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        ), "val_metric"
    return None, "none"


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------
def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None,
              grad_clip=None, desc="train"):
    """Run one pass. If optimizer is None, runs in eval (no-grad) mode."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, all_true, all_pred = 0.0, [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=desc, leave=False):
            images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            use_amp = scaler is not None
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, labels)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_true.extend(labels.detach().cpu().numpy().tolist())
            all_pred.extend(logits.argmax(1).detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    qwk, macro_f1 = quick_metrics(np.array(all_true), np.array(all_pred))
    return {"loss": avg_loss, "qwk": qwk, "macro_f1": macro_f1}


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train(cfg, model_name=None, smoke_test=False):
    if model_name:
        cfg["model"]["name"] = model_name
    name = cfg["model"]["name"]

    set_seed(cfg["project"]["seed"])
    device = get_device()
    reports_dir = Path(cfg["paths"]["reports_dir"]); reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "logs").mkdir(exist_ok=True)
    logger = get_logger("train", reports_dir / "logs" / f"train_{name}.log")
    logger.info(f"Training '{name}' | smoke_test={smoke_test} | device={device}")

    # --- data ---
    preprocess = build_preprocess(cfg)
    loaders = build_dataloaders(
        cfg,
        train_transform=build_train_transforms(cfg),
        eval_transform=build_eval_transforms(cfg),
        preprocess=preprocess,
    )
    train_loader, val_loader = loaders["train"], loaders["val"]
    class_weights = loaders["class_weights"]
    logger.info(f"train batches={len(train_loader)} val batches={len(val_loader)}")

    # --- smoke test: shrink to 1 epoch and a couple of batches ---
    epochs = cfg["train"]["epochs"]
    if smoke_test:
        epochs = 1
        train_loader = _truncate_loader(train_loader, 2)
        val_loader = _truncate_loader(val_loader, 2)
        logger.info("SMOKE TEST: 1 epoch, 2 batches each.")

    # --- model / loss / optim / sched ---
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg, class_weights, device)
    optimizer = build_optimizer(cfg, model)
    scheduler, sched_step = build_scheduler(cfg, optimizer)
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # --- early stopping setup ---
    monitor = cfg["train"].get("early_stop_metric", "qwk")
    patience = cfg["train"].get("early_stop_patience", 7)
    best_score, best_epoch, epochs_no_improve = -np.inf, -1, 0
    history = {"train_loss": [], "val_loss": [], "val_qwk": [], "val_macro_f1": []}

    models_dir = Path(cfg["paths"]["models_dir"]); models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / f"{name}_best.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler,
                       grad_clip=cfg["train"].get("grad_clip"), desc=f"epoch {epoch} train")
        va = run_epoch(model, val_loader, loss_fn, device, desc=f"epoch {epoch} val")

        # scheduler step
        if scheduler is not None:
            if sched_step == "val_metric":
                scheduler.step(va[monitor])
            else:
                scheduler.step()

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["val_qwk"].append(va["qwk"])
        history["val_macro_f1"].append(va["macro_f1"])

        score = va[monitor] if monitor != "val_loss" else -va["loss"]
        improved = score > best_score
        logger.info(
            f"epoch {epoch}/{epochs} | {time.time()-t0:.0f}s | "
            f"train_loss {tr['loss']:.3f} | val_loss {va['loss']:.3f} | "
            f"val_qwk {va['qwk']:.3f} | val_macroF1 {va['macro_f1']:.3f}"
            + ("  *best*" if improved else "")
        )

        if improved:
            best_score, best_epoch, epochs_no_improve = score, epoch, 0
            save_checkpoint(
                {
                    "model_name": name,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_score": best_score,
                    "monitor": monitor,
                    "config": cfg,
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and not smoke_test:
                logger.info(f"Early stopping at epoch {epoch} (no improvement in {patience}).")
                break

    # --- save history + curves ---
    with open(reports_dir / f"history_{name}.json", "w") as f:
        json.dump(history, f, indent=2)
    if len(history["train_loss"]) >= 1:
        plot_training_curves(history, Path(cfg["paths"]["figures_dir"]) / f"training_curves_{name}.png")
    logger.info(f"Done. Best {monitor}={best_score:.4f} at epoch {best_epoch}. Checkpoint: {ckpt_path}")
    return {"best_score": best_score, "best_epoch": best_epoch, "checkpoint": str(ckpt_path)}


def _truncate_loader(loader, n_batches):
    """Yield only the first n_batches (used for the smoke test)."""
    from itertools import islice

    class _Wrap:
        def __init__(self, dl, n):
            self.dl, self.n, self.dataset = dl, n, dl.dataset
        def __iter__(self):
            return islice(iter(self.dl), self.n)
        def __len__(self):
            return min(self.n, len(self.dl))

    return _Wrap(loader, n_batches)


def main():
    ap = argparse.ArgumentParser(description="Train a DR classifier.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None, help="override model.name from config")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs")
    ap.add_argument("--smoke-test", action="store_true", help="1 epoch, 2 batches")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    train(cfg, model_name=args.model, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
