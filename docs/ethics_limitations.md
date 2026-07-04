# Ethics, limitations, and future work

> **This is a research and education project, not a medical device.** It makes
> no clinical claims and must not be used to diagnose, screen, or treat anyone.
> The disclaimer in the README and app is not boilerplate — it is the single
> most important statement in this repository.

## Ethics

**Not a diagnostic tool; clinician-in-the-loop is essential.** Even a model with
strong test metrics has not been validated for clinical use. Any real screening
deployment requires a qualified clinician to review and own the decision. This
project is a demonstration of the *machine-learning lifecycle*, not a product.

**Dataset and demographic bias.** APTOS 2019 comes from a specific set of clinics
and cameras in India. A model trained on it can encode the biases of that
distribution — skin/retinal pigmentation ranges, camera optics, disease
prevalence — and underperform on populations it never saw. Fairness across
demographic groups cannot be assumed and is not measured here.

**Automation bias.** When a tool outputs a confident grade (and a plausible
Grad-CAM heatmap), humans tend to defer to it — even when it is wrong. A
well-placed heatmap is *not* evidence of correct reasoning. Any interface must
make the model's role advisory and its uncertainty visible, and must resist
nudging clinicians toward uncritical acceptance.

**Privacy.** Retinal images are medical data. Real use must follow applicable
privacy law and institutional rules (consent, de-identification, secure storage,
access control). This project ships **no** patient data and gitignores `data/`.

**Regulatory reality.** Autonomous DR-screening software is a regulated medical
device in many jurisdictions (e.g. FDA-cleared systems exist for DR). Shipping
something like this for clinical use without the corresponding clinical
validation and regulatory clearance would be unsafe and, in most places, illegal.

## Limitations

- **Single-source data.** Trained and tested on one dataset; no external or
  multi-site validation. Reported metrics likely overstate real-world performance.
- **Label noise.** Public DR labels contain grader disagreement and errors; the
  model can only be as good as its labels.
- **Generalization gaps.** Different cameras, resolutions, lighting, and patient
  populations can degrade performance substantially.
- **No external clinical validation.** Test metrics are on a held-out split of the
  *same* dataset, not on independent clinical data.
- **Image-quality dependence.** Blurred, off-center, or poorly exposed images
  degrade predictions; the pipeline flags but does not fix bad inputs.
- **Class imbalance.** Severe/proliferative grades are rare, so their metrics are
  estimated from few examples and are noisier.
- **Explainability is coarse.** Grad-CAM shows correlation, not causation, and can
  look convincing on wrong predictions.

## Future work

- **More and more-diverse data**, with **external validation** on independent
  clinical datasets and explicit per-subgroup fairness analysis.
- **Ordinal regression / ordinal-aware losses** that model the 0–4 ordering
  directly, instead of treating grades as unordered classes.
- **Lesion segmentation** to localize microaneurysms/hemorrhages/exudates, which
  is more clinically interpretable than a coarse heatmap.
- **Model ensembles** for accuracy and calibration.
- **Uncertainty estimation** (e.g. MC-dropout, deep ensembles) to flag
  low-confidence cases for human review rather than forcing a grade.
- **Image-quality gating** to reject ungradable images up front.
- **Prospective, clinician-in-the-loop evaluation** before any thought of
  deployment, plus the regulatory pathway that would entail.
