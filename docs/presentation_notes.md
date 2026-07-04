# Presentation notes & FAQ

Talking points to present this project confidently, plus a Q&A you can rehearse.
Pair this with `docs/concepts.md` (plain-language definitions of every term).

## The one-sentence pitch
"I built a reproducible, end-to-end deep-learning pipeline that grades retinal
photos for diabetic retinopathy, compares a from-scratch CNN against three
transfer-learning models, explains its predictions with Grad-CAM, and serves it
all through a polished web app — as a learning project, not a medical device."

## Slide-by-slide outline

1. **Title.** Project name + the one-sentence pitch + the non-diagnostic disclaimer.
2. **The problem.** DR is a leading cause of preventable blindness and is
   screenable from a single fundus photo → a good, high-impact ML target.
3. **The data.** APTOS 2019, ~3,662 images, ICDR grades 0–4. Show the class
   distribution: ~49% "No DR" → heavy imbalance is the central challenge.
4. **The grading scale.** Explain ICDR 0–4 is *ordinal*, and "referable DR"
   (grade ≥ 2) as the decision-relevant binary view.
5. **Preprocessing.** Show the before/after figure (FOV crop → Ben-Graham →
   resize). One line on *why* Ben-Graham makes lesions pop.
6. **Augmentation.** Show examples; emphasize "gentle on purpose" so we don't
   erase tiny lesions.
7. **Models.** Custom CNN (baseline, ~0.39M params) vs ResNet50 / EfficientNet /
   DenseNet. One sentence each on residuals / compound scaling / dense connectivity.
8. **Training.** Class-weighted loss, AdamW, cosine LR, early stopping, AMP. Same
   engine + config for everyone → fair comparison.
9. **Results.** The comparison table. Lead with **QWK**, not accuracy, and explain
   why accuracy is misleading under imbalance. Show confusion matrices.
10. **Explainability.** Grad-CAM gallery (correct + a few errors). State the
    caution: heatmaps show correlation, not proof.
11. **The app.** Live demo or screenshots: upload → grade + probabilities →
    Grad-CAM. Point out the persistent disclaimer.
12. **Ethics & limitations.** Bias, automation bias, no external validation,
    regulation. Be the person who names the limitations first.
13. **Future work + close.** Ordinal regression, lesion segmentation, uncertainty,
    external validation. Restate: research/education, not a device.

## FAQ to rehearse

**Q: Why Quadratic Weighted Kappa instead of accuracy?**
Because the data is imbalanced and the grades are ordinal. Accuracy rewards a lazy
"always No-DR" model; QWK measures agreement corrected for chance and penalizes
far-off errors (4-vs-0) more than near ones (1-vs-0), matching clinical severity.

**Q: Why does transfer learning beat your custom CNN?**
The pretrained backbones already learned generic visual features from 1.2M
ImageNet images, so they need far less data to specialize. The custom CNN starts
from random weights with ~0.39M params — a useful baseline, but outmatched.

**Q: How did you handle the class imbalance?**
Class-weighted CrossEntropy (rare grades weighted up), optionally a
WeightedRandomSampler to balance batches, and focal loss as an alternative. And I
evaluate with imbalance-robust metrics (macro-F1, per-class recall, QWK).

**Q: Is the comparison fair?**
Yes — every model uses the same data splits, preprocessing, augmentation, loss,
optimizer, and config. Only the architecture changes.

**Q: What does Grad-CAM actually show, and can I trust it?**
It highlights image regions that most influenced the predicted grade
(gradients of the class score w.r.t. the last conv feature maps). A trustworthy
DR heatmap sits on lesions. But it shows correlation, not clinical reasoning, and
can look convincing on a wrong prediction — so it's a sanity check, not proof.

**Q: Can this be used in a clinic?**
No. It has no external clinical validation and is not a regulated medical device.
Real DR-screening software requires clinical validation, regulatory clearance, and
a clinician in the loop. This is a learning project.

**Q: How is it reproducible?**
Pinned dependencies, a fixed seed applied everywhere (`config.yaml` →
`utils.set_seed`), one config driving every script, and a documented run order.
A fresh clone + the dataset reproduces the results.

**Q: Why 224×224 and not 512?**
224 matches the pretrained backbones, trains fast, and fits modest GPUs. 512 can
improve detection of tiny lesions but costs ~5× compute — a documented trade-off,
easy to change in `config.yaml`.

**Q: What would you do with more time?**
Ordinal-aware modeling, lesion segmentation for better interpretability,
uncertainty estimates to defer hard cases to humans, and external validation on a
different dataset to test real generalization.
