"""Model loading and prediction for the demo.

Five models, one entry point (predict_all):

  logreg    - Logistic regression, 6 global features       (models/logreg_global6.joblib)
  rf6       - Random Forest,       6 global features       (models/rf_global6_demo.joblib)
  rf160     - Random Forest,     160 classical features    (models/rf160_demo.joblib)
  scratch   - CNN from scratch,  128 px pixels             (models/scratch_cnn.keras)
  densenet  - DenseNet-121,      128 px pixels, ImageNet   (models/densenet121_inference.keras)

DenseNet codes against models/inference_spec.json (written by notebook 13);
the classical models against models/classical_spec.json (notebook 14).
Input convention for both CNNs: float32, raw 0-255, RGB, 128x128 -
preprocessing lives INSIDE the model graph. Class order = class_names.json.

Nothing is trained here. Nothing is downloaded here. Every file is loaded
once, cached, and MD5-verified against models/MD5SUMS.txt first, so the
wrong file can never fail silently.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import streamlit as st

from config import MODEL_ORDER, MODEL_INFO

# repo root = one level above streamlit/ - works locally and on Streamlit Cloud
REPO_ROOT   = Path(__file__).resolve().parent.parent
MODELS_DIR  = REPO_ROOT / "models"
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
IMG_SIZE    = 128  # asserted against inference_spec.json in load_bundle()


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _expected_md5s() -> dict:
    """Parse models/MD5SUMS.txt -> {filename: md5}. Comment lines ignored."""
    out = {}
    for line in (MODELS_DIR / "MD5SUMS.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def _verify(*files: Path):
    """Stop the app if any listed file's MD5 differs from MD5SUMS.txt."""
    expected = _expected_md5s()
    for f in files:
        if not f.exists():
            st.error(f"Missing model file: models/{f.name}. Stopping.")
            st.stop()
        exp = expected.get(f.name)
        if exp is not None and _md5(f) != exp:
            st.error(
                f"MD5 mismatch for {f.name} - this is not the verified file. "
                "Re-download it from the repo / Drive export. Stopping."
            )
            st.stop()


# --------------------------------------------------------------------------
# DenseNet-121 (unchanged - gradcam_utils.py depends on this exact function)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the DenseNet-121 model (once)...")
def load_bundle():
    """Verify MD5s, load spec + class names + model. Cached across reruns."""
    spec        = json.loads((MODELS_DIR / "inference_spec.json").read_text())
    class_names = json.loads((MODELS_DIR / "class_names.json").read_text())
    model_file  = MODELS_DIR / spec["model_file"]

    _verify(model_file, MODELS_DIR / "class_names.json")

    assert spec["img_size"] == IMG_SIZE, "inference_spec img_size != 128"
    assert len(class_names) == spec["n_classes"] == 38

    import tensorflow as tf  # deferred: keeps start-up snappy
    model = tf.keras.models.load_model(model_file, compile=False)
    return model, class_names, spec


def predict(pil_image, top_k: int = 3):
    """DenseNet only. PIL image -> [(class_name, probability), ...] top-k."""
    model, class_names, _ = load_bundle()
    probs = _cnn_probs(model, pil_image)
    idx   = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in idx]


# --------------------------------------------------------------------------
# Scratch CNN
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the scratch CNN (once)...")
def load_scratch_cnn():
    model_file = MODELS_DIR / "scratch_cnn.keras"
    _verify(model_file)
    import tensorflow as tf
    model = tf.keras.models.load_model(model_file, compile=False)
    assert tuple(model.input_shape[1:]) == (IMG_SIZE, IMG_SIZE, 3), \
        f"scratch CNN expects {model.input_shape}, app feeds {IMG_SIZE}px"
    assert model.output_shape[-1] == 38
    return model


def _cnn_probs(model, pil_image) -> np.ndarray:
    """Both CNNs: RGB, resize to 128x128, float32 raw 0-255 - NO manual
    preprocessing, the model graph does it internally."""
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x   = np.asarray(img, dtype="float32")[None, ...]      # (1,128,128,3), 0-255
    return model.predict(x, verbose=0)[0]


# --------------------------------------------------------------------------
# Classical models (LogReg, RF-6, RF-160)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the classical models (once)...")
def load_classical():
    """-> (spec, {key: fitted sklearn estimator}) for logreg / rf6 / rf160."""
    import joblib
    spec  = json.loads((MODELS_DIR / "classical_spec.json").read_text())
    files = {k: MODELS_DIR / spec["models"][k]["file"] for k in ("logreg", "rf6", "rf160")}
    _verify(*files.values())
    models = {k: joblib.load(f) for k, f in files.items()}
    for k, m in models.items():
        assert len(m.classes_) == 38, f"{k}: expected 38 classes, got {len(m.classes_)}"
    return spec, models


def classical_features(pil_image, file_size_kb: float):
    """-> (global6 vector (6,), feat160 vector (160,)) for one image."""
    from features import feat160, global6
    spec, _ = load_classical()
    return global6(pil_image, file_size_kb), feat160(pil_image, file_size_kb, spec["feat160"])


def _sk_probs(model, x) -> np.ndarray:
    """sklearn predict_proba -> full 38-vector in canonical class order."""
    p = model.predict_proba(x[None, :])[0]
    out = np.zeros(38, dtype="float64")
    out[np.asarray(model.classes_, dtype=int)] = p
    return out


# --------------------------------------------------------------------------
# One call for the demo page
# --------------------------------------------------------------------------
def predict_all(pil_image, file_size_kb: float, which=None):
    """Run the selected models on one image.

    Returns a list of dicts (one per model, in MODEL_ORDER):
      key, name, sees, val_f1, top (class_name), conf, top3 [(class, p), ...]
    """
    which = list(MODEL_ORDER) if which is None else [k for k in MODEL_ORDER if k in which]
    _, class_names, _ = load_bundle()

    probs = {}
    if any(k in which for k in ("logreg", "rf6", "rf160")):
        spec, models = load_classical()
        g6, f160 = classical_features(pil_image, file_size_kb)
        if "logreg" in which:
            probs["logreg"] = _sk_probs(models["logreg"], g6)
        if "rf6" in which:
            probs["rf6"] = _sk_probs(models["rf6"], g6)
        if "rf160" in which:
            probs["rf160"] = _sk_probs(models["rf160"], f160)
    if "scratch" in which:
        probs["scratch"] = _cnn_probs(load_scratch_cnn(), pil_image)
    if "densenet" in which:
        probs["densenet"] = _cnn_probs(load_bundle()[0], pil_image)

    rows = []
    for k in which:
        p   = probs[k]
        idx = np.argsort(p)[::-1][:3]
        rows.append({
            "key":    k,
            "name":   MODEL_INFO[k]["name"],
            "sees":   MODEL_INFO[k]["sees"],
            "val_f1": _val_f1(k),
            "top":    class_names[idx[0]],
            "conf":   float(p[idx[0]]),
            "top3":   [(class_names[i], float(p[i])) for i in idx],
        })
    return rows


def _val_f1(key: str) -> float:
    """Validation macro-F1 of the model that is actually loaded.
    CNNs: from config (Deliverable 2). Classical: from classical_spec.json
    (the demo copies report their own number, not the report's)."""
    if key in ("logreg", "rf6", "rf160"):
        spec, _ = load_classical()
        return float(spec["models"][key]["val_macro_f1"])
    return float(MODEL_INFO[key]["val_f1"])


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def list_samples():
    """Committed demo images -> [(path, true_class_from_filename), ...]."""
    out = []
    for p in sorted(SAMPLES_DIR.glob("sample_*")):
        true_cls = p.stem.split("_", 2)[2]                 # sample_01_<class>
        out.append((p, true_cls))
    return out


def pretty(cls: str) -> str:
    """'Tomato___Late_blight' -> 'Tomato - Late blight' for display."""
    crop, _, cond = cls.partition("___")
    return f"{crop.replace('_', ' ')} - {cond.replace('_', ' ')}" if cond else cls
