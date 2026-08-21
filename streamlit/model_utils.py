"""Model loading and prediction for the demo.

Codes against models/inference_spec.json — the contract written by
notebook 13 (export). Input convention: float32, raw 0-255, RGB, 128x128,
preprocessing lives INSIDE the model graph. Class order = class_names.json.

Nothing is trained here. Nothing is downloaded here. The model is loaded
once from the committed file and cached; its MD5 is verified against
models/MD5SUMS.txt first, so the wrong file can never fail silently.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import streamlit as st

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


@st.cache_resource(show_spinner="Loading the DenseNet-121 model (once)...")
def load_bundle():
    """Verify MD5s, load spec + class names + model. Cached across reruns."""
    spec        = json.loads((MODELS_DIR / "inference_spec.json").read_text())
    class_names = json.loads((MODELS_DIR / "class_names.json").read_text())
    model_file  = MODELS_DIR / spec["model_file"]

    expected = _expected_md5s()
    for f in (model_file, MODELS_DIR / "class_names.json"):
        exp = expected.get(f.name)
        if exp is not None and _md5(f) != exp:
            st.error(
                f"MD5 mismatch for {f.name} - this is not the verified file. "
                "Re-download it from the repo / Drive export. Stopping."
            )
            st.stop()

    assert spec["img_size"] == IMG_SIZE, "inference_spec img_size != 128"
    assert len(class_names) == spec["n_classes"] == 38

    import tensorflow as tf  # deferred: keeps non-demo sections snappy
    model = tf.keras.models.load_model(model_file, compile=False)
    return model, class_names, spec


def predict(pil_image, top_k: int = 3):
    """PIL image -> [(class_name, probability), ...] top-k, best first.

    Follows inference_spec.json exactly: RGB, resize to 128x128,
    float32 kept on the raw 0-255 scale - NO manual preprocessing,
    the model graph does it internally.
    """
    model, class_names, _ = load_bundle()
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x   = np.asarray(img, dtype="float32")[None, ...]      # (1,128,128,3), 0-255
    probs = model.predict(x, verbose=0)[0]
    idx   = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in idx]


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
