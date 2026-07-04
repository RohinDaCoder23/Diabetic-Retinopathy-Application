"""
custom_cnn.py — a small convolutional neural network built FROM SCRATCH.

Purpose: a teaching baseline. By assembling the network by hand we can explain
exactly what every layer does, and we get a fair "no pretrained weights"
reference point to compare against transfer learning.

Architecture (input 3 x 224 x 224):

    Block      Layers                                  Output shape (C,H,W)
    --------   -------------------------------------   --------------------
    input                                              3   x 224 x 224
    block1     Conv(3->32) BN ReLU, MaxPool(2)         32  x 112 x 112
    block2     Conv(32->64) BN ReLU, MaxPool(2)        64  x  56 x  56
    block3     Conv(64->128) BN ReLU, MaxPool(2)       128 x  28 x  28
    block4     Conv(128->256) BN ReLU, MaxPool(2)      256 x  14 x  14
    head       GlobalAvgPool -> 256                    256
               Dropout -> Linear(256->num_classes)     num_classes

What each layer is for (plain language; full glossary in docs/concepts.md):
    * Conv2d    : slides learnable filters to detect local patterns (edges,
                  textures, then lesion-like shapes deeper in the network).
    * BatchNorm : normalizes each layer's outputs, which stabilizes and speeds
                  up training.
    * ReLU      : the nonlinearity (max(0,x)); lets the net learn complex,
                  non-linear relationships.
    * MaxPool   : halves spatial size, keeping the strongest responses; gives
                  translation tolerance and cuts compute.
    * Global Average Pooling : averages each channel over space -> one number
                  per channel. Replaces a huge flatten+dense, so far fewer
                  parameters and less overfitting.
    * Dropout   : randomly zeroes activations during training to reduce
                  overfitting.
    * Linear    : the final classifier mapping features -> 5 grade scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool. The repeated building block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # halves H and W
        )

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    """A compact 4-block CNN with a global-average-pooled classifier head."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.3, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),   # -> 32 x 112 x 112
            ConvBlock(32, 64),            # -> 64 x 56 x 56
            ConvBlock(64, 128),           # -> 128 x 28 x 28
            ConvBlock(128, 256),          # -> 256 x 14 x 14
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> 256 x 1 x 1 (size-agnostic)
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # -> 256
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),  # -> num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    @property
    def gradcam_target_layer(self):
        """The last conv layer — the natural target for Grad-CAM."""
        return self.features[-1].block[0]


def build_custom_cnn(num_classes: int = 5, dropout: float = 0.3) -> nn.Module:
    """Factory used by train.py so every model is created the same way."""
    return CustomCNN(num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    # Smoke test: build the model, run a dummy batch, print shapes + param count.
    model = build_custom_cnn(num_classes=5)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[custom_cnn] trainable parameters: {n_params:,}")

    dummy = torch.randn(4, 3, 224, 224)  # batch of 4 fake images
    out = model(dummy)
    print(f"[custom_cnn] input {tuple(dummy.shape)} -> output {tuple(out.shape)}")
    assert out.shape == (4, 5), "Output should be (batch, num_classes)"
    print("[custom_cnn] forward pass OK")
