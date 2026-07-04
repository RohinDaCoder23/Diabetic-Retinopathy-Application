"""
streamlit_app.py — Diabetic Retinopathy Detector (research/education demo).

Run from the repo root:
    streamlit run app/streamlit_app.py

A polished, single-file Streamlit app:
    Upload a fundus image -> pick a model -> Analyze -> predicted grade +
    confidence + probability bar chart + Grad-CAM overlay.

*** NOT A MEDICAL DEVICE. Research/education only. No clinical claims. ***
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Make `src` importable whether run from repo root or app/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Page config (wide layout, emoji favicon) — must be the first st call.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetic Retinopathy Detector",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Small, tasteful CSS: calm medical palette, rounded cards, clean spacing.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
      .hero {
        background: linear-gradient(135deg, #2A9D8F 0%, #264653 100%);
        color: #fff; padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
      }
      .hero h1 { color:#fff; margin:0; font-size:1.9rem; }
      .hero p  { color:#e8f4f2; margin:.35rem 0 0; font-size:1rem; }
      .card {
        background:#fff; border:1px solid #e6ebec; border-radius:14px;
        padding:1.1rem 1.25rem; box-shadow:0 1px 3px rgba(0,0,0,.04); margin-bottom:1rem;
      }
      .pred-grade { font-size:2.1rem; font-weight:700; color:#264653; margin:0; }
      .pred-sub   { color:#5b6b73; margin:.1rem 0 0; }
      .disclaimer {
        background:#FFF4E5; border:1px solid #F2C879; color:#7a5a12;
        border-radius:10px; padding:.7rem 1rem; font-size:.9rem; margin-bottom:1rem;
      }
      .muted { color:#6b7a80; font-size:.9rem; }
      .stProgress > div > div > div { background-color:#2A9D8F; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Config + model metadata
# ---------------------------------------------------------------------------
import yaml  # noqa: E402

CONFIG_PATH = ROOT / "config.yaml"
cfg = yaml.safe_load(open(CONFIG_PATH))
CLASS_NAMES = cfg["classes"]["names"]
REFERABLE = cfg["classes"]["referable_threshold"]

MODEL_INFO = {
    "custom_cnn":      "Custom CNN — small from-scratch baseline (~0.39M params).",
    "resnet50":        "ResNet50 — deep residual network; robust transfer-learning baseline.",
    "efficientnet_b0": "EfficientNet-B0 — best accuracy-per-parameter via compound scaling.",
    "efficientnet_b3": "EfficientNet-B3 — larger EfficientNet for a bit more accuracy.",
    "densenet121":     "DenseNet121 — dense connectivity; parameter-efficient feature reuse.",
}


@st.cache_resource(show_spinner=False)
def get_model(model_name: str):
    """Load + cache the model so it isn't rebuilt on every interaction."""
    from src.inference import load_model
    return load_model(cfg, model_name)


def disclaimer():
    st.markdown(
        '<div class="disclaimer">⚠️ <b>Not a medical device.</b> This tool is for '
        "research and education only. It does <b>not</b> provide a diagnosis. "
        "Always consult a qualified clinician.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    from src.inference import list_available_checkpoints
    available = list_available_checkpoints(cfg)

    model_name = st.selectbox(
        "Model",
        options=list(MODEL_INFO.keys()),
        format_func=lambda k: ("✅ " if k in available else "⬜ ") + k,
        help="✅ = trained weights found. ⬜ = no checkpoint yet (train it first).",
    )
    st.caption(MODEL_INFO[model_name])
    if model_name not in available:
        st.warning("No trained weights for this model yet. See **How it works** below.")

    st.divider()
    with st.expander("ℹ️ About"):
        st.write(
            "Grades retinal fundus photos for diabetic retinopathy on the ICDR "
            "0–4 scale, and explains predictions with Grad-CAM. A research/education "
            "project — **not** a diagnostic tool."
        )
    with st.expander("🛠️ How it works"):
        st.markdown(
            "1. Upload a fundus photo.\n"
            "2. The image is preprocessed exactly as in training (field-of-view "
            "crop → Ben-Graham normalization → resize → ImageNet normalize).\n"
            "3. The selected model predicts a grade + probabilities.\n"
            "4. Grad-CAM highlights the regions that drove the prediction.\n\n"
            "**No weights?** Train a model on Google Colab (see `docs/colab_quickstart.md`) "
            "and place `models/<name>_best.pt` in the repo."
        )
    with st.expander("⚠️ Limitations"):
        st.markdown(
            "- Trained on a single dataset (APTOS 2019); may not generalize to "
            "other cameras/populations.\n"
            "- No external clinical validation.\n"
            "- Grad-CAM shows correlation, not clinical proof — a confident heatmap "
            "can sit on a wrong prediction.\n"
            "- Sensitive to image quality."
        )

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>🩺 Diabetic Retinopathy Detector</h1>'
    "<p>Upload a retinal fundus photo to estimate diabetic retinopathy severity "
    "(ICDR grade 0–4), with an explainable Grad-CAM overlay.</p></div>",
    unsafe_allow_html=True,
)
disclaimer()

# ---------------------------------------------------------------------------
# Upload + empty state
# ---------------------------------------------------------------------------
col_up, col_tip = st.columns([2, 1])
with col_up:
    uploaded = st.file_uploader(
        "Upload a retinal fundus image",
        type=["png", "jpg", "jpeg"],
        help="Drag & drop a fundus photograph (PNG/JPG).",
    )
with col_tip:
    st.markdown(
        '<div class="card"><b>No image handy?</b><br>'
        '<span class="muted">Tick the box below to analyze a built-in sample '
        "image so you can see the full flow.</span></div>",
        unsafe_allow_html=True,
    )
    use_sample = st.checkbox("Use sample image")

# Resolve the input image (uploaded or sample).
image_rgb = None
source_label = None
if uploaded is not None:
    from PIL import Image, UnidentifiedImageError
    try:
        image_rgb = np.array(Image.open(uploaded).convert("RGB"))
        source_label = uploaded.name
    except UnidentifiedImageError:
        st.error("That file doesn't look like a valid image. Please upload a PNG or JPG.")
elif use_sample:
    sample_path = ROOT / "app" / "assets" / "sample_fundus.png"
    if sample_path.exists():
        from PIL import Image
        image_rgb = np.array(Image.open(sample_path).convert("RGB"))
        source_label = "sample_fundus.png"
    else:
        st.info("Sample image not found at app/assets/sample_fundus.png.")

if image_rgb is None:
    st.markdown(
        '<div class="card"><h4>👋 How to start</h4>'
        '<span class="muted">Upload a fundus image (or tick “Use sample image”), '
        "pick a model in the sidebar, then click <b>Analyze</b>. You'll get a "
        "predicted grade, a probability chart across all five grades, and a "
        "Grad-CAM heatmap of where the model looked.</span></div>",
        unsafe_allow_html=True,
    )
    st.stop()

# Preview + analyze button
st.markdown("#### Preview")
pcol1, pcol2 = st.columns([1, 2])
with pcol1:
    st.image(image_rgb, caption=f"Input: {source_label}", use_column_width=True)
with pcol2:
    st.markdown(
        '<div class="card"><span class="muted">Ready to analyze with '
        f"<b>{model_name}</b>. The image will be preprocessed exactly as during "
        "training before inference.</span></div>",
        unsafe_allow_html=True,
    )
    analyze = st.button("🔬 Analyze", type="primary", use_container_width=True)

if not analyze:
    st.stop()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
if model_name not in available:
    st.error(
        f"**No trained weights for `{model_name}`.** Train it first "
        "(see `docs/colab_quickstart.md`) and place the checkpoint at "
        f"`models/{model_name}_best.pt`, or pick a model marked ✅ in the sidebar."
    )
    st.stop()

try:
    progress = st.progress(0, text="Loading model…")
    with st.spinner("Loading model…"):
        model, device = get_model(model_name)
    progress.progress(35, text="Preprocessing image…")

    from src.inference import preprocess_image, predict, explain
    tensor, clean = preprocess_image(image_rgb, cfg)

    progress.progress(60, text="Running inference…")
    pred, probs = predict(model, tensor, device)

    progress.progress(85, text="Generating Grad-CAM…")
    try:
        rgb01, overlay = explain(model, tensor, model_name, pred, cfg)
    except Exception as e:  # Grad-CAM is best-effort; never block the prediction
        rgb01, overlay = None, None
        gradcam_error = str(e)
    progress.progress(100, text="Done")
    progress.empty()
except Exception as e:
    st.error(f"Something went wrong during analysis: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
st.markdown("### Results")
confidence = float(probs[pred]) * 100
referable = pred >= REFERABLE

r1, r2 = st.columns(2)
with r1:
    st.markdown(
        f'<div class="card"><p class="pred-grade">{CLASS_NAMES[pred]}</p>'
        f'<p class="pred-sub">Predicted ICDR grade</p></div>',
        unsafe_allow_html=True,
    )
    m1, m2 = st.columns(2)
    m1.metric("Confidence", f"{confidence:.1f}%")
    m2.metric("Referable (grade ≥ 2)", "Yes" if referable else "No")
    if referable:
        st.info("Model suggests **referable** DR — in a real setting this would prompt clinician review.")
with r2:
    st.markdown("**Probability across all grades**")
    import pandas as pd
    prob_df = pd.DataFrame({"grade": CLASS_NAMES, "probability": probs}).set_index("grade")
    st.bar_chart(prob_df, height=240, color="#2A9D8F")

st.divider()

# Original vs Grad-CAM
st.markdown("### Explainability — where the model looked")
g1, g2 = st.columns(2)
with g1:
    st.image(clean, caption="Preprocessed input", use_column_width=True)
with g2:
    if overlay is not None:
        st.image(overlay, caption="Grad-CAM overlay (hot = more influential)", use_column_width=True)
    else:
        st.warning("Grad-CAM could not be generated for this model/image.")
st.caption(
    "Grad-CAM shows correlation with the prediction, not clinical proof. A "
    "well-placed heatmap can still accompany a wrong prediction — interpret with care."
)
disclaimer()
