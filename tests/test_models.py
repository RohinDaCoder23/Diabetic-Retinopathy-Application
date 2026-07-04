"""Model-builder tests. Need torch; pretrained=False so no network."""
from __future__ import annotations

import pytest


def _base_cfg(name):
    return {
        "model": {"name": name, "pretrained": False, "dropout": 0.3, "freeze_backbone": False},
        "classes": {"num_classes": 5},
    }


@pytest.mark.needs_torch
def test_custom_cnn_param_count_matches_docs():
    pytest.importorskip("torch")
    from src.models.custom_cnn import build_custom_cnn

    model = build_custom_cnn(num_classes=5)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # This exact number is quoted in the README/handoff; lock it down.
    assert n == 390_181


@pytest.mark.needs_torch
@pytest.mark.parametrize("name", ["custom_cnn", "resnet50", "efficientnet_b0", "densenet121"])
def test_build_model_forward_shape(name):
    torch = pytest.importorskip("torch")
    from src.models import build_model

    model = build_model(_base_cfg(name)).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5), f"{name} should output (batch, num_classes)"


@pytest.mark.needs_torch
def test_unknown_model_raises():
    pytest.importorskip("torch")
    from src.models import build_model

    with pytest.raises(ValueError):
        build_model(_base_cfg("not_a_real_model"))


@pytest.mark.needs_torch
@pytest.mark.parametrize("name", ["custom_cnn", "resnet50", "densenet121"])
def test_gradcam_target_layer_available(name):
    pytest.importorskip("torch")
    from src.models import build_model, get_gradcam_target_layer

    model = build_model(_base_cfg(name))
    layer = get_gradcam_target_layer(model, name)
    assert layer is not None


@pytest.mark.needs_torch
def test_freeze_backbone_only_trains_head():
    pytest.importorskip("torch")
    from src.models import build_model

    cfg = _base_cfg("resnet50")
    cfg["model"]["freeze_backbone"] = True
    model = build_model(cfg)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    # All trainable params should be in the classifier head (fc / classifier).
    assert trainable, "something should still be trainable (the head)"
    assert all(n.startswith(("fc", "classifier")) for n in trainable)
