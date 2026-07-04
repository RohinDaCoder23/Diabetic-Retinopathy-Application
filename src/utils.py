"""
utils.py — shared helpers used across the whole project.

This module deliberately has NO heavy logic. It holds the small, reusable
plumbing that every script needs:

    * set_seed()        -> make runs reproducible
    * load_config()     -> read config.yaml into a plain dict
    * get_device()      -> pick GPU if available, else CPU
    * get_logger()      -> consistent console + file logging
    * save_checkpoint() -> persist model weights + metadata
    * load_checkpoint() -> restore them
    * plot helpers      -> training curves, confusion matrices

WHY a separate utils module? Because reproducibility, logging, and
checkpointing are concerns shared by training, evaluation, Grad-CAM, and the
app. Keeping them in one place means there is exactly one definition of
"set the seed" or "load a checkpoint", so the whole project behaves the same
way everywhere.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Torch / matplotlib are imported lazily inside functions where possible so
# that lightweight callers (e.g. a quick config check) don't pay the import
# cost. But seeding genuinely needs torch, so we import it at module load.
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every random number generator we use.

    Deep-learning pipelines draw randomness from several places: Python's
    ``random``, NumPy, and PyTorch (including the CUDA backend). If any of
    them is unseeded, two "identical" runs can diverge. We seed all of them.

    Args:
        seed: the integer seed (kept in config.yaml so it's version-controlled).
        deterministic: if True, ask cuDNN to use deterministic algorithms.
            This trades a little speed for exact reproducibility — worth it for
            a research/education project where you want repeatable numbers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make hash-based ops (rare, but e.g. set ordering) reproducible too.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # benchmark=True is faster but nondeterministic


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker process.

    Each DataLoader worker is a separate process with its own RNG state. Pass
    this as ``worker_init_fn`` so augmentations are reproducible across workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load config.yaml into a nested dict.

    We keep config as plain YAML (not Python) so non-programmers can tweak
    settings, and so a config is easy to diff in git.
    """
    import yaml  # imported here so callers that don't need YAML stay light

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at '{path}'. Run scripts from the repo root, "
            f"or pass --config with the correct path."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device(verbose: bool = True) -> "torch.device":
    """Return the best available device.

    Order of preference: CUDA GPU -> Apple Metal (MPS) -> CPU. On Colab you'll
    normally get a CUDA GPU. On a Mac laptop you may get MPS. Everything in this
    project runs on CPU too, just slowly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
        name = "Apple MPS"
    else:
        device = torch.device("cpu")
        name = "CPU"
    if verbose:
        print(f"[device] Using: {device} ({name})")
    return device


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "dr", log_file: Optional[str | Path] = None) -> logging.Logger:
    """Create a logger that prints to console and (optionally) to a file.

    Consistent logging beats scattered print() calls: every line gets a
    timestamp and level, and we can save the full run log to disk for the
    record.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured — avoid duplicate handlers
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(
    state: Dict[str, Any],
    path: str | Path,
) -> None:
    """Save a checkpoint dict to disk.

    A "checkpoint" is more than weights — we also store the epoch, optimizer
    state, the config used, and the best metric, so training can be resumed and
    so we know exactly how a model was produced.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | None = None) -> Dict[str, Any]:
    """Load a checkpoint dict from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


# ---------------------------------------------------------------------------
# Plotting helpers (kept here so notebooks and scripts share one style)
# ---------------------------------------------------------------------------
def plot_training_curves(history: Dict[str, list], out_path: str | Path) -> None:
    """Plot train/val loss and the tracked metric vs. epoch, and save to disk.

    Args:
        history: dict with keys like 'train_loss', 'val_loss', 'val_qwk'.
        out_path: where to save the PNG.
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Loss
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss vs epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    # Metric (whatever was tracked, e.g. QWK or macro-F1)
    metric_keys = [k for k in history if k.startswith("val_") and k != "val_loss"]
    for k in metric_keys:
        axes[1].plot(history[k], label=k)
    axes[1].set_title("Validation metric vs epoch")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, out_path: str | Path, title: str = "Confusion matrix") -> None:
    """Render a confusion matrix heatmap and save it.

    Args:
        cm: 2D array (rows = true, cols = predicted).
        class_names: list of label strings for the axes.
        out_path: where to save the PNG.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def count_parameters(model) -> int:
    """Return the number of trainable parameters in a model.

    Handy for the README and for comparing model sizes fairly.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Tiny self-test so you can run `python src/utils.py` and see it works.
    set_seed(42)
    print("[utils] seed set OK")
    get_device()
    cfg = load_config("config.yaml") if Path("config.yaml").exists() else {}
    print(f"[utils] config loaded with top-level keys: {list(cfg.keys())}")
