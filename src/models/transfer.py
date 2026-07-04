"""
transfer.py — transfer-learning model builders.

"Transfer learning" = start from a network already trained on ImageNet (1.2M
natural images across 1000 classes) and adapt it to our 5-class DR task. The
early layers already encode generic visual features (edges, textures, shapes),
so we reuse them and only re-learn the final decision layer (feature
extraction) or fine-tune the whole network.

Two modes, controlled by `freeze_backbone` in config.yaml:
    * Feature extraction (freeze_backbone: true): freeze the pretrained body,
      train only the new classifier head. Fast, needs little data, good first
      pass.
    * Fine-tuning (freeze_backbone: false): train everything end-to-end at a
      small learning rate. Usually higher accuracy; the default here.

Recommended recipe: feature-extract for a few epochs to warm up the head, then
unfreeze and fine-tune. `set_backbone_trainable()` lets train.py do this.

Architecture notes (full versions in docs/concepts.md):
    ResNet50      : residual "skip" connections add the input of a block to its
                    output, so gradients flow through very deep nets without
                    vanishing — that's what made 50+ layer training practical.
    EfficientNet  : "compound scaling" grows depth, width, and input resolution
                    together in a principled ratio, hitting high accuracy with
                    far fewer parameters/FLOPs (B0 small, B3 bigger).
    DenseNet121   : every layer receives the feature maps of ALL previous layers
                    (dense connectivity), encouraging feature reuse and strong
                    gradient flow with relatively few parameters.
"""

from __future__ import annotations

import torch.nn as nn


# ---------------------------------------------------------------------------
# Builders. Each loads ImageNet weights (if pretrained=True) and swaps the
# classifier head for a `num_classes`-way head with dropout.
# ---------------------------------------------------------------------------
def _weights(pretrained: bool, weights_enum):
    """Return the torchvision weights enum (new API) or None."""
    return weights_enum.DEFAULT if pretrained else None


def build_resnet50(num_classes=5, pretrained=True, dropout=0.3) -> nn.Module:
    from torchvision import models

    m = models.resnet50(weights=_weights(pretrained, models.ResNet50_Weights))
    in_feat = m.fc.in_features  # 2048
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, num_classes))
    m._gradcam_target = m.layer4[-1]  # last residual block for Grad-CAM
    return m


def build_efficientnet_b0(num_classes=5, pretrained=True, dropout=0.3) -> nn.Module:
    from torchvision import models

    m = models.efficientnet_b0(weights=_weights(pretrained, models.EfficientNet_B0_Weights))
    in_feat = m.classifier[1].in_features  # 1280
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, num_classes))
    m._gradcam_target = m.features[-1]
    return m


def build_efficientnet_b3(num_classes=5, pretrained=True, dropout=0.3) -> nn.Module:
    from torchvision import models

    m = models.efficientnet_b3(weights=_weights(pretrained, models.EfficientNet_B3_Weights))
    in_feat = m.classifier[1].in_features  # 1536
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, num_classes))
    m._gradcam_target = m.features[-1]
    return m


def build_densenet121(num_classes=5, pretrained=True, dropout=0.3) -> nn.Module:
    from torchvision import models

    m = models.densenet121(weights=_weights(pretrained, models.DenseNet121_Weights))
    in_feat = m.classifier.in_features  # 1024
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, num_classes))
    m._gradcam_target = m.features.norm5  # final BN before the head
    return m


# Registry so the rest of the code can build any model from its name string.
TRANSFER_BUILDERS = {
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "efficientnet_b3": build_efficientnet_b3,
    "densenet121": build_densenet121,
}


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze every parameter EXCEPT the classifier head.

    Used for feature-extraction (trainable=False) vs fine-tuning (True), and to
    implement the warm-up-then-unfreeze recipe.
    """
    head_names = ("fc", "classifier")
    for name, param in model.named_parameters():
        is_head = any(name.startswith(h) for h in head_names)
        param.requires_grad = trainable or is_head
