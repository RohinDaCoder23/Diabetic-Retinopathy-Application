"""src package — pipeline modules for the diabetic retinopathy project.

Import order of responsibilities:
    utils      -> seeds, config, logging, checkpoints, plots
    preprocess -> deterministic image cleanup (FOV crop, Ben-Graham, CLAHE)
    augment    -> Albumentations train/val transforms
    data       -> Dataset + DataLoaders + stratified split + sampler
    models/    -> custom_cnn + transfer-learning builders
    train      -> the training loop
    evaluate   -> metrics + figures
    gradcam    -> explainability heatmaps
"""

__version__ = "0.1.0"
