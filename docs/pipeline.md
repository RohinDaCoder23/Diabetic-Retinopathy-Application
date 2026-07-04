# Pipeline — data → preprocess → train → eval → explain → serve

This document is the end-to-end map of how a fundus image becomes a prediction,
with a short **WHY** note for each modeling choice. It fills in as we build.

## Overview (the run order)

```
data/ (APTOS drop-in)
   │
   ▼
[1] EDA              src + notebooks/01_eda.ipynb
   │   understand class balance, image quality, splits
   ▼
[2] Preprocess      src/preprocess.py
   │   FOV crop → Ben-Graham → (CLAHE) → resize → normalize
   ▼
[3] Augment         src/augment.py
   │   train-only random flips/rotations/brightness
   ▼
[4] Data loaders    src/data.py
   │   stratified split + (optional) balanced sampler
   ▼
[5] Train           src/train.py
   │   weighted CE loss, AdamW, scheduler, AMP, early stop
   ▼
[6] Evaluate        src/evaluate.py
   │   accuracy, P/R/F1, ROC-AUC, confusion matrix, QWK
   ▼
[7] Explain         src/gradcam.py
   │   Grad-CAM heatmaps over fundus images
   ▼
[8] Serve           app/streamlit_app.py
       upload → predict → probabilities → Grad-CAM
```

## WHY notes

### Preprocessing (deterministic — `src/preprocess.py`)
- **Field-of-view crop** — Fundus photos are a small off-center circle on a big
  black canvas. Cropping to the retina spends our limited resolution on actual
  tissue and makes images from different cameras more comparable.
- **Ben-Graham normalization** (`4*img - 4*blur + 128`) — Cameras vary wildly in
  color/lighting. Subtracting a heavily-blurred copy removes slow lighting
  gradients and amplifies high-frequency detail, making microaneurysms and
  hemorrhages far more visible. (This trick won the 2015 Kaggle DR competition.)
- **CLAHE** (optional, off by default) — Local contrast boost that can make faint
  lesions pop, applied on the LAB luminance channel so colors aren't distorted.
  Overlaps with Ben-Graham, so we don't run both by default.
- **Resize 224** — Matches ImageNet-pretrained backbones; cheap and fast. Bump to
  384/512 later for finer lesions if GPU budget allows.
- **ImageNet normalization** — Pretrained weights expect inputs normalized with
  ImageNet mean/std, so we must match it. Kept in `augment.py` so train and eval
  pipelines share exactly one definition.

### Augmentation (random, train-only — `src/augment.py`)
- **Flips + small rotations** — A retina has no canonical orientation, so these
  are realistic and safe; they multiply effective data for free.
- **Mild brightness/contrast + small shift/zoom** — Mimics camera and exposure
  variation so the model generalizes across devices.
- **Deliberately gentle** — DR grade hinges on lesions a few pixels wide.
  Aggressive crops/zoom/warps can erase or fabricate that evidence and teach the
  wrong thing. Robustness, not distortion.
- **Eval transforms are deterministic** (normalize + to-tensor only) so
  validation/test numbers are stable and comparable.

### Class-imbalance strategy (`src/data.py`, configured in `config.yaml`)
- **Why it matters** — ~49% of images are "No DR". A model can score ~50%
  accuracy by always predicting healthy while missing every sick patient — the
  dangerous failure mode. We never trust accuracy alone (see metrics in M6).
- **Weighted CrossEntropy** (`compute_class_weights`, inverse-frequency,
  normalized to mean 1) — makes rare grades count as much as common ones.
- **WeightedRandomSampler** (`make_weighted_sampler`) — oversamples rare grades
  so each batch is roughly balanced. Use instead of, or with, weighted loss.
- **Focal loss** (alternative, `focal_gamma`) — down-weights easy examples to
  focus learning on hard ones; wired into the loss in M4.
- **Guidance** — Start with weighted loss alone; combining all three over-corrects.
