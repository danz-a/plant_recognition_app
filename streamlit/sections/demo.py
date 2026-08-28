"""Live Demo - one leaf, five models, weakest first.

Two paths: the committed sample gallery (the safe, rehearsed path) and a
free upload. The models are loaded once, cached, and never trained here.
"""
import streamlit as st
from PIL import Image

from components import section_header
from config import MODEL_INFO, MODEL_ORDER
from gradcam_utils import gradcam_overlay
from model_utils import (classical_features, list_samples, load_bundle,
                         load_classical, load_scratch_cnn, predict_all, pretty)
from features import GLOBAL6_COLS


def _model_picker():
    st.sidebar.header("Models on stage")
    which = st.sidebar.multiselect(
        "Models",
        MODEL_ORDER,
        default=MODEL_ORDER,
        format_func=lambda k: MODEL_INFO[k]["name"],
        label_visibility="collapsed",
    )
    st.sidebar.caption(
        "Weakest first. Remove models to walk up the ladder one step at a time; "
        "the table keeps the same order."
    )
    return which


def _warm(which):
    """Load the selected models on first visit so the prediction is instant."""
    load_bundle()
    if any(k in which for k in ("logreg", "rf6", "rf160")):
        load_classical()
    if "scratch" in which:
        load_scratch_cnn()


def _show(image, file_size_kb, which, true_cls=None):
    if not which:
        st.info("Pick at least one model in the sidebar.")
        return

    col_img, col_res = st.columns([1, 1.7])

    with col_img:
        st.image(image, width="stretch")
        if "densenet" in which:
            cam = gradcam_overlay(image)
            if cam is not None:
                overlay, _, _ = cam
                st.image(
                    overlay,
                    caption="DenseNet-121 Grad-CAM - where the final model looked "
                            "(heat = evidence for its predicted class; 4x4 grid, upsampled)",
                    width="stretch",
                )
            else:
                st.caption("Grad-CAM unavailable for this image - predictions are unaffected.")

    with col_res:
        if true_cls is not None:
            st.markdown(f"**True class:** {pretty(true_cls)}")

        rows = predict_all(image, file_size_kb, which)

        lines = ["| Model | Sees | Val macro-F1 | Prediction | Confidence | |",
                 "|---|---|:---:|---|:---:|:---:|"]
        for r in rows:
            mark = "" if true_cls is None else ("✅" if r["top"] == true_cls else "❌")
            lines.append(
                f"| **{r['name']}** | {r['sees']} | {r['val_f1']:.3f} "
                f"| {pretty(r['top'])} | {r['conf']:.1%} | {mark} |"
            )
        st.markdown("\n".join(lines))

        # top-3 of one model (DenseNet by default - the confusion-pair story lives there)
        names = {r["key"]: r["name"] for r in rows}
        default = "densenet" if "densenet" in names else rows[-1]["key"]
        focus = st.selectbox("Top-3 of", list(names), index=list(names).index(default),
                             format_func=names.get)
        for cls, p in next(r for r in rows if r["key"] == focus)["top3"]:
            st.progress(min(max(p, 0.0), 1.0), text=f"{pretty(cls)} - {p:.1%}")

        if any(k in which for k in ("logreg", "rf6")):
            with st.expander("The six numbers the 6-feature models see"):
                g6, _ = classical_features(image, file_size_kb)
                st.table({c: [f"{v:.2f}"] for c, v in zip(GLOBAL6_COLS, g6)})
                st.caption(
                    "file_size_kb is a property of the JPEG, not of the leaf - "
                    "the shortcut feature discussed in the modelling report. "
                    "Six whole-image averages cannot see a lesion; that is why "
                    "these two models sit at the bottom of the table."
                )


def render():
    section_header(
        "Live Demo",
        "A photograph goes in - crop and condition come out. "
        "Five models from the project, weakest first, all loaded from verified "
        "checkpoints; nothing is trained at runtime.",
    )

    which = _model_picker()
    _warm(which)

    tab_samples, tab_upload = st.tabs(["Sample images (test split)", "Upload an image"])

    with tab_samples:
        samples = list_samples()
        if not samples:
            st.warning("No images found in data/samples/.")
        else:
            labels = [pretty(t) for _, t in samples]
            choice = st.radio("Choose a leaf", labels, horizontal=True,
                              label_visibility="collapsed")
            path, true_cls = samples[labels.index(choice)]
            st.caption(
                "These images come from the sealed test split - "
                "none of the models has ever trained on them."
            )
            _show(Image.open(path), path.stat().st_size / 1024, which, true_cls)

    with tab_upload:
        up = st.file_uploader("Leaf photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if up is not None:
            _show(Image.open(up), up.size / 1024, which)
        else:
            st.caption(
                "PlantVillage-style images (single leaf, plain background) "
                "play to the models' strengths - field photos are the known, "
                "openly discussed limitation."
            )
