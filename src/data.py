"""
data.py — labels, stratified splits, Dataset, DataLoaders, imbalance helpers.

DESIGN NOTE (why some imports are lazy):
    The label-reading and splitting functions only need pandas + scikit-learn,
    so they're importable without PyTorch. This lets the EDA notebook reuse the
    exact same split logic the training code uses, without dragging in torch.
    The Dataset / DataLoader / sampler functions import torch *inside* the
    function, so they only require torch when you actually train.

The on-disk layout this module expects (APTOS 2019) is documented in
data/README.md:

    data/aptos2019/
        train.csv            # columns: id_code, diagnosis
        train_images/<id_code>.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ===========================================================================
# 1. Labels
# ===========================================================================
def read_labels(cfg: dict) -> pd.DataFrame:
    """Read the labels CSV and attach a resolved image path to each row.

    Returns a DataFrame with columns:
        id_code   : the image id (filename stem)
        diagnosis : the integer ICDR grade 0..4
        path      : absolute/relative path to the image file

    We verify the image files exist up front so problems surface early with a
    clear message, rather than deep inside the training loop.
    """
    csv_path = Path(cfg["paths"]["train_csv"])
    img_dir = Path(cfg["paths"]["train_images"])
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Labels CSV not found: {csv_path}\n"
            f"Download APTOS 2019 into data/ — see data/README.md."
        )
    df = pd.read_csv(csv_path)
    if not {"id_code", "diagnosis"}.issubset(df.columns):
        raise ValueError(
            f"Expected columns 'id_code' and 'diagnosis' in {csv_path}, "
            f"got {list(df.columns)}"
        )

    # APTOS images are <id_code>.png. If your copy uses another extension,
    # change the suffix here (single point of truth).
    df = df.copy()
    df["path"] = df["id_code"].apply(lambda x: str(img_dir / f"{x}.png"))

    missing = [p for p in df["path"].tolist() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image file(s) referenced in {csv_path} are missing "
            f"under {img_dir}. First few: {missing[:3]}"
        )
    return df


def class_distribution(df: pd.DataFrame, num_classes: int) -> pd.Series:
    """Return counts per grade 0..num_classes-1 (including zeros for absent grades)."""
    counts = df["diagnosis"].value_counts().sort_index()
    return counts.reindex(range(num_classes), fill_value=0)


# ===========================================================================
# 2. Stratified train / val / test split
# ===========================================================================
def make_splits(
    df: pd.DataFrame,
    cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / val / test, stratified by grade.

    WHY stratified? APTOS is imbalanced (most images are "No DR"). A plain
    random split could, by chance, leave very few rare-grade (e.g. Severe)
    images in validation or test, making those metrics noisy or undefined.
    Stratifying forces each split to keep the same class proportions as the
    full dataset.

    The split is reproducible because it uses the fixed seed from config.yaml.
    We split off the test set first, then carve val out of the remainder, so
    the test fraction is measured against the whole dataset.
    """
    from sklearn.model_selection import train_test_split  # lazy: sklearn only

    seed = cfg["project"]["seed"]
    val_size = cfg["split"]["val_size"]
    test_size = cfg["split"]["test_size"]
    stratify_on = df["diagnosis"] if cfg["split"].get("stratify", True) else None

    # Step 1: hold out the test set.
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=stratify_on
    )

    # Step 2: carve validation out of the remaining train+val portion.
    # Recompute the val fraction relative to the train_val subset.
    val_fraction_of_remainder = val_size / (1.0 - test_size)
    stratify_tv = train_val_df["diagnosis"] if stratify_on is not None else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction_of_remainder,
        random_state=seed,
        stratify=stratify_tv,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ===========================================================================
# 3. Imbalance helpers
# ===========================================================================
def compute_class_weights(train_df: pd.DataFrame, num_classes: int):
    """Inverse-frequency class weights for a weighted CrossEntropy loss.

    Idea: rare classes get a bigger weight so the loss "cares" about them as
    much as the common "No DR" class. We normalize so the weights average to 1
    (keeps the loss magnitude comparable to the unweighted case).

    Returns a torch.FloatTensor of length num_classes.
    """
    import torch  # lazy: only needed when training

    counts = class_distribution(train_df, num_classes).values.astype(np.float64)
    counts = np.clip(counts, 1.0, None)  # avoid divide-by-zero if a class is absent
    weights = counts.sum() / (num_classes * counts)  # inverse-frequency, balanced
    weights = weights / weights.mean()  # normalize to mean 1.0
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(train_df: pd.DataFrame, num_classes: int):
    """Build a WeightedRandomSampler that oversamples rare grades.

    Alternative (or complement) to weighted loss: instead of re-weighting the
    loss, re-weight how often each image is *drawn*. Each sample's draw
    probability is inversely proportional to its class frequency, so a training
    batch ends up roughly class-balanced.

    Use weighted loss OR this sampler OR both — controlled in config.yaml under
    `imbalance`. Using both can over-correct, so try one first.
    """
    from torch.utils.data import WeightedRandomSampler  # lazy

    counts = class_distribution(train_df, num_classes).values.astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    per_class_weight = 1.0 / counts
    sample_weights = train_df["diagnosis"].map(
        {c: per_class_weight[c] for c in range(num_classes)}
    ).values
    sample_weights = sample_weights.astype(np.float64)
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,  # required for oversampling
    )


# ===========================================================================
# 4. Dataset
# ===========================================================================
def _load_image_rgb(path: str) -> np.ndarray:
    """Load an image as an RGB uint8 numpy array (H, W, 3).

    We use OpenCV for speed but convert BGR->RGB so colors are correct for both
    display and the ImageNet-normalized models.
    """
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_dataset_class():
    """Factory that builds the APTOSDataset class (keeps torch import lazy).

    Returns the class. Most callers will instead use `build_dataloaders`, but
    this is handy in notebooks where you want a Dataset without DataLoaders.
    """
    from torch.utils.data import Dataset

    class APTOSDataset(Dataset):
        """PyTorch Dataset for APTOS fundus images.

        Args:
            df: DataFrame with columns id_code, diagnosis, path.
            preprocess: callable(np.uint8 HWC RGB) -> np.uint8 HWC RGB.
                Deterministic cleanup (FOV crop, Ben-Graham...). From M2.
            transform: Albumentations transform applied AFTER preprocess; it
                must output a CHW float tensor (e.g. via ToTensorV2). From M2.
            return_id: if True, also return the id_code (useful for Grad-CAM /
                error analysis).
        """

        def __init__(self, df, preprocess=None, transform=None, return_id=False):
            self.df = df.reset_index(drop=True)
            self.preprocess = preprocess
            self.transform = transform
            self.return_id = return_id

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = _load_image_rgb(row["path"])
            if self.preprocess is not None:
                img = self.preprocess(img)
            if self.transform is not None:
                img = self.transform(image=img)["image"]  # Albumentations API
            label = int(row["diagnosis"])
            if self.return_id:
                return img, label, row["id_code"]
            return img, label

    return APTOSDataset


# ===========================================================================
# 5. DataLoaders
# ===========================================================================
def build_dataloaders(
    cfg: dict,
    train_transform=None,
    eval_transform=None,
    preprocess=None,
) -> Dict[str, object]:
    """Build train/val/test DataLoaders from config.

    Wires together: read_labels -> make_splits -> Dataset -> DataLoader, with
    the optional WeightedRandomSampler when `imbalance.use_sampler` is true.

    Returns a dict with keys: train, val, test (DataLoaders), plus class_weights
    (tensor) and the three DataFrames for inspection.
    """
    from torch.utils.data import DataLoader

    from .utils import seed_worker  # reproducible workers

    num_classes = cfg["classes"]["num_classes"]
    df = read_labels(cfg)
    train_df, val_df, test_df = make_splits(df, cfg)

    APTOSDataset = make_dataset_class()
    train_ds = APTOSDataset(train_df, preprocess, train_transform)
    val_ds = APTOSDataset(val_df, preprocess, eval_transform)
    test_ds = APTOSDataset(test_df, preprocess, eval_transform, return_id=True)

    use_sampler = cfg["imbalance"].get("use_sampler", False)
    sampler = make_weighted_sampler(train_df, num_classes) if use_sampler else None

    common = dict(
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("num_workers", 2),
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(
        train_ds,
        shuffle=(sampler is None),  # if sampler is used, it handles ordering
        sampler=sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_weights": compute_class_weights(train_df, num_classes),
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


# ===========================================================================
# 6. Synthetic demo data (FOR TESTING / DEMO ONLY — never for real results)
# ===========================================================================
def generate_synthetic_dataset(
    out_dir: str | Path,
    n_per_class: Optional[List[int]] = None,
    image_size: int = 256,
    seed: int = 42,
) -> Path:
    """Create a tiny, FAKE APTOS-like dataset so the pipeline can be exercised
    before the real data is downloaded.

    *** This generates random fundus-LOOKING images. It is purely for smoke-
    testing the code path (EDA, loaders, a 1-epoch training run). Any "results"
    on synthetic data are meaningless — do not present them. ***

    It writes:
        out_dir/train.csv               (id_code, diagnosis)
        out_dir/train_images/*.png

    Default class counts mimic APTOS's imbalance shape (lots of grade 0).
    """
    import cv2

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    img_dir = out_dir / "train_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    if n_per_class is None:
        n_per_class = [40, 8, 22, 5, 7]  # ~APTOS proportions, scaled down

    rows = []
    counter = 0
    for grade, n in enumerate(n_per_class):
        for _ in range(n):
            # Build a fundus-like image: dark background + bright circular retina.
            img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            center = (image_size // 2, image_size // 2)
            radius = int(image_size * 0.45)
            base = rng.integers(60, 110)
            color = (int(base * 0.4), int(base * 0.5), int(base))  # reddish (BGR)
            cv2.circle(img, center, radius, color, -1)
            # Add "lesion"-like specks; more for higher grades (purely cosmetic).
            for _ in range(grade * 8):
                x = rng.integers(center[0] - radius, center[0] + radius)
                y = rng.integers(center[1] - radius, center[1] + radius)
                if (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius ** 2:
                    cv2.circle(img, (int(x), int(y)), rng.integers(1, 4),
                               (40, 40, 220), -1)  # red-ish specks
            id_code = f"synthetic_{counter:05d}"
            cv2.imwrite(str(img_dir / f"{id_code}.png"), img)
            rows.append({"id_code": id_code, "diagnosis": grade})
            counter += 1

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(out_dir / "train.csv", index=False)
    return out_dir


if __name__ == "__main__":
    # Quick self-test using synthetic data (no real download needed).
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    generate_synthetic_dataset(tmp / "aptos2019", seed=0)
    cfg = {
        "paths": {
            "train_csv": str(tmp / "aptos2019" / "train.csv"),
            "train_images": str(tmp / "aptos2019" / "train_images"),
        },
        "classes": {"num_classes": 5},
        "split": {"val_size": 0.15, "test_size": 0.15, "stratify": True},
        "project": {"seed": 42},
    }
    df = read_labels(cfg)
    tr, va, te = make_splits(df, cfg)
    print(f"[data] synthetic total={len(df)} | train={len(tr)} val={len(va)} test={len(te)}")
    print("[data] class distribution:\n", class_distribution(df, 5))
