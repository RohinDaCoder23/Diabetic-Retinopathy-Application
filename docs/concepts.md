# Concepts — plain-language glossary

This file explains, in plain English, every concept used in the project so you
can confidently present it. Start here if a term in the code or README is unfamiliar.

> How to use this when presenting: read the **one-line** version aloud, then use
> the **why it matters here** sentence to connect it to our DR project.

---

## The medical problem

**Diabetic retinopathy (DR).** A complication of diabetes where high blood
sugar damages the tiny blood vessels in the retina (the light-sensitive tissue
at the back of the eye). Left undetected it can cause blindness. It's
*screenable*: a photo of the retina can reveal early damage before symptoms
appear — which is exactly what makes it a good machine-learning target.

**Fundus photograph.** A photo of the inside back surface of the eye (the
"fundus"), including the retina, optic disc, and blood vessels. Our model's
input. It looks like an orange-red circle on a black background.

**Lesions that signal DR** (what a model — and a doctor — looks for):
- *Microaneurysms*: tiny red dots, the earliest visible sign.
- *Hemorrhages*: larger blot-shaped bleeds.
- *Hard exudates*: bright yellow deposits (leaked fats/proteins).
- *Neovascularization*: abnormal new vessels (the "proliferative", grade-4 stage).

**ICDR 0–4 grading scale.** The International Clinical Diabetic Retinopathy
severity scale we predict: 0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4
Proliferative. It is **ordinal** — the grades have a meaningful order and
spacing, so being "off by one" is a smaller error than being "off by four".

**Referable DR (grade ≥ 2).** A clinically useful *binary* simplification:
patients at moderate-or-worse generally need referral to an eye specialist. We
report this alongside the full 5-class view because it maps to a real decision.

---

## Machine-learning basics

**Image classification.** Given an image, output a label. Here: given a fundus
photo, output one of 5 DR grades.

**Convolutional Neural Network (CNN).** The standard neural network for images.
It slides small learnable filters ("convolutions") across the image to detect
patterns — edges first, then textures, then lesion-like shapes in deeper layers.

**Training / validation / test split.**
- *Training set*: the model learns from these.
- *Validation set*: used during training to tune choices and decide when to
  stop. The model never learns directly from it.
- *Test set*: touched **once**, at the very end, to report honest performance.
Keeping these separate prevents fooling ourselves (a model that memorized the
training images would look great on them but fail on new patients).

**Stratified split.** Splitting so each subset keeps the same class proportions
as the whole dataset. Important here because most images are "No DR" — a naive
random split could leave too few rare-grade images in validation/test.

**Class imbalance.** When some classes are far more common than others (APTOS is
~half "No DR"). A lazy model can score high accuracy by always guessing the
majority class while completely missing sick patients — the dangerous failure.
We counter it with **class-weighted loss** and/or a **balanced sampler**
(details in `docs/pipeline.md`).

**Epoch / batch.** An *epoch* is one full pass over the training data. A *batch*
is the small group of images processed at once (e.g. 32). We update the model
once per batch.

**Loss function.** A number measuring how wrong the model is; training tries to
minimize it. We use **cross-entropy** (standard for classification), weighted by
class frequency to respect the imbalance.

**Reproducibility / random seed.** Neural-network training uses randomness
(weight init, shuffling, augmentation). Fixing a *seed* makes those random draws
repeatable, so the same code gives the same result. We set it in `config.yaml`
and apply it in `src/utils.py`.

---

## Preprocessing & augmentation (M2)

### Preprocessing (deterministic — same at train/val/test/inference)
- **Field-of-view crop.** Trims the black border so the retina fills the frame,
  focusing the model (and our limited resolution) on actual tissue.
- **Ben-Graham normalization.** `4·img − 4·blur + 128`. Subtracting a blurred
  copy removes lighting/color differences between cameras and makes tiny lesions
  pop. (Named after the 2015 Kaggle DR winner.)
- **CLAHE** (optional). Contrast-Limited Adaptive Histogram Equalization boosts
  *local* contrast on the luminance channel; off by default as it overlaps with
  Ben-Graham.
- **Resize** to 224×224 — matches ImageNet-pretrained backbones.
- **ImageNet normalization.** Subtract the ImageNet mean and divide by its std
  per channel, because pretrained weights expect inputs in that exact range.

### Augmentation (random — training only)
- **What.** Horizontal/vertical flips, small rotations (±20°), mild
  brightness/contrast, small shift/zoom — re-rolled every epoch.
- **Why.** Free extra variety → better generalization. Fundus images have no
  canonical orientation, so flips/rotations are realistic; mild photometric
  jitter mimics different cameras.
- **Why gentle.** DR grade depends on lesions a few pixels wide; aggressive
  crops/zoom/warps could erase or fabricate that evidence. Validation/test use
  no randomness (normalize + to-tensor only) so scores stay comparable.

## Transfer learning (M5)

- **Pretraining.** The backbones (ResNet50, EfficientNet, DenseNet) were first
  trained on ImageNet (1.2M natural images). They already encode generic
  features — edges, textures, shapes — that transfer to fundus images.
- **Feature extraction vs fine-tuning.** *Feature extraction* freezes the
  pretrained body and trains only the new 5-class head (fast, little data).
  *Fine-tuning* trains everything at a small learning rate (usually higher
  accuracy). A common recipe: extract first to warm up the head, then unfreeze
  and fine-tune. Controlled by `freeze_backbone` in `config.yaml`.
- **Why these three architectures.**
  - **ResNet50 — residual connections.** Each block adds its input to its output
    (a "skip"), so gradients flow through 50+ layers without vanishing. That's
    what made very deep networks trainable. ~25M params.
  - **EfficientNet — compound scaling.** Instead of arbitrarily making a net
    deeper or wider, it scales depth, width, and input resolution together in a
    fixed ratio, hitting high accuracy with far fewer params/FLOPs. B0 is small;
    B3 is bigger.
  - **DenseNet121 — dense connectivity.** Every layer receives the feature maps
    of *all* previous layers, encouraging feature reuse and strong gradient flow
    with relatively few parameters (~8M).
- **Fair comparison.** All models — including the custom CNN — are trained with
  the *same* engine, config, data splits, preprocessing, and loss, so
  differences reflect the architecture, not the setup.

### The custom CNN, layer by layer (M3)

We build a small CNN by hand so every piece is explainable. Input is a
`3 × 224 × 224` image (3 color channels). The "feature extractor" is four
repeated blocks; the "head" turns features into 5 grade scores.

```
input            3   × 224 × 224
block1  Conv 3→32   + BatchNorm + ReLU + MaxPool   →  32 × 112 × 112
block2  Conv 32→64  + BatchNorm + ReLU + MaxPool   →  64 ×  56 ×  56
block3  Conv 64→128 + BatchNorm + ReLU + MaxPool   → 128 ×  28 ×  28
block4  Conv 128→256+ BatchNorm + ReLU + MaxPool   → 256 ×  14 ×  14
head    GlobalAvgPool → 256 → Dropout → Linear(256→5)  →  5 scores
```

**Trainable parameters: ~390,181** (≈0.39M) — tiny next to ResNet50's ~25M,
which is the point: it's a lightweight, from-scratch baseline.

- **Convolution (Conv2d).** Slides small learnable filters across the image to
  detect local patterns. Early layers learn edges/colors; deeper layers combine
  those into textures and lesion-like shapes. We increase channels (32→256) as
  spatial size shrinks, so the network trades "where" for "what".
- **BatchNorm.** Re-centers/rescales each layer's outputs per batch, which keeps
  the signal well-behaved, allows higher learning rates, and speeds convergence.
- **ReLU.** `max(0, x)` — the nonlinearity that lets stacked layers represent
  complex functions (without it, the whole net collapses to a linear map).
- **MaxPool(2).** Keeps the strongest response in each 2×2 region, halving height
  and width. Adds small-shift tolerance and cuts compute.
- **Global Average Pooling.** Averages each channel over all spatial positions →
  one number per channel (256 numbers). Replaces a massive flatten+dense layer,
  slashing parameters and overfitting, and works at any input size.
- **Dropout.** Randomly zeroes activations during training so the model can't
  rely on any single feature — a simple, strong regularizer.
- **Linear head.** Maps the 256 features to 5 numbers ("logits"), one per grade;
  softmax later turns them into probabilities.

## Training concepts (M4)

- **Loss = class-weighted CrossEntropy.** CrossEntropy measures how far the
  predicted probabilities are from the truth; weighting by inverse class
  frequency stops the model ignoring rare severe grades. **Focal loss** is an
  alternative that further down-weights easy examples.
- **Optimizer = AdamW.** Adam adapts the step size per parameter (fast, little
  tuning); the "W" applies weight decay correctly for better regularization.
- **Learning-rate scheduler.** *Cosine annealing* smoothly lowers the LR over
  training to settle into a good minimum; *ReduceLROnPlateau* drops it when
  validation stalls.
- **Early stopping.** Stop when the validation metric (QWK) hasn't improved for
  N epochs — saves compute and avoids overfitting.
- **Mixed precision (AMP).** Uses 16-bit math on GPU for speed and lower memory,
  with negligible accuracy cost; auto-disabled on CPU.
- **Gradient clipping.** Caps the gradient size to keep training stable.

### Evaluation metrics (M6)

- **Accuracy.** Fraction correct. *Misleading here:* with ~49% "No DR", always
  guessing 0 scores ~49% while catching zero patients. We report it but never
  lead with it.
- **Precision / Recall / F1 (per class + macro).** Precision = "of images I
  called grade k, how many were?"; Recall = "of true grade-k images, how many
  did I find?". F1 is their harmonic mean. *Macro* averages classes equally, so
  rare severe grades count as much as common ones. **Recall on severe grades is
  the clinically important number** — missing a sick patient is the costly error.
- **Confusion matrix.** Grid of true (rows) vs predicted (cols). The diagonal is
  correct; off-diagonal cells show *which* grades get mixed up (usually adjacent
  ones, e.g. 1↔2).
- **ROC-AUC (one-vs-rest, macro).** Threshold-independent measure of how well the
  model separates each grade from the rest; 0.5 = chance, 1.0 = perfect.
- **Quadratic Weighted Kappa (QWK) — our headline metric.** Measures agreement
  with the true grade, *corrected for chance*, and weights errors by squared
  distance, so predicting 4-vs-0 hurts far more than 1-vs-0. This respects the
  ordinal 0–4 scale and is the standard DR-grading metric (used by the Kaggle
  competition). Roughly: <0.4 poor, 0.6–0.8 good, >0.8 excellent.
- **Referable-DR recall (grade ≥ 2).** The binary "should this patient be
  referred?" view — a direct, decision-relevant summary number.
### Grad-CAM — seeing where the model looked (M7)

**What it is.** Grad-CAM (Gradient-weighted Class Activation Mapping) produces a
heatmap over the input image showing which regions most influenced the model's
prediction. Hot regions pushed the predicted-grade score up the most.

**How it's computed** (plain language):
1. Run the image forward and pick the target class (usually the prediction).
2. Backpropagate that class's score to the **last convolutional layer's** feature
   maps (these still carry spatial layout — "where").
3. Average each feature map's gradients into a single weight = "how much this map
   mattered for this class".
4. Take the weighted sum of the feature maps, keep the positive part (ReLU),
   upscale to image size, and overlay it as a colored heatmap.

**How to read it.** A *trustworthy* DR heatmap concentrates on **lesions** —
microaneurysms, hemorrhages, exudates — not on the black border or empty retina.

**Caution (important for your presentation).** Grad-CAM is a *coarse* localizer
and only shows what correlates with the output — it is **not** proof of correct
clinical reasoning. A confident, nicely-placed heatmap can still accompany a
**wrong** prediction. Treat it as a sanity check and teaching aid, never as
evidence the model is medically right. This is doubly important because this
project is **not** a medical device.
