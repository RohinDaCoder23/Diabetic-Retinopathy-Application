"""models package — one factory to build any architecture from config.

    custom_cnn.py -> a small CNN built from scratch (baseline / teaching model)
    transfer.py   -> ImageNet-pretrained backbones (ResNet50, EfficientNet, DenseNet)

`build_model(cfg)` reads `config.yaml -> model.name` and returns the right
nn.Module, so train.py / evaluate.py / the app never hard-code an architecture.
`get_gradcam_target_layer(model, name)` returns the conv layer Grad-CAM hooks.
"""

from __future__ import annotations

VALID_MODELS = [
    "custom_cnn",
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b3",
    "densenet121",
]


def build_model(cfg: dict):
    """Build a model from the config's `model` section.

    Reads: model.name, model.pretrained, model.dropout, classes.num_classes,
           model.freeze_backbone.
    """
    name = cfg["model"]["name"]
    num_classes = cfg["classes"]["num_classes"]
    dropout = cfg["model"].get("dropout", 0.3)
    pretrained = cfg["model"].get("pretrained", True)

    if name not in VALID_MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose one of {VALID_MODELS}.")

    if name == "custom_cnn":
        from .custom_cnn import build_custom_cnn
        # The custom CNN is always trained from scratch (no ImageNet weights).
        return build_custom_cnn(num_classes=num_classes, dropout=dropout)

    from .transfer import TRANSFER_BUILDERS, set_backbone_trainable
    model = TRANSFER_BUILDERS[name](
        num_classes=num_classes, pretrained=pretrained, dropout=dropout
    )
    if cfg["model"].get("freeze_backbone", False):
        set_backbone_trainable(model, trainable=False)  # feature-extraction mode
    return model


def get_gradcam_target_layer(model, name: str):
    """Return the convolutional layer Grad-CAM should attach to."""
    if name == "custom_cnn":
        return model.gradcam_target_layer
    if hasattr(model, "_gradcam_target"):
        return model._gradcam_target
    raise ValueError(f"No Grad-CAM target registered for model '{name}'.")
