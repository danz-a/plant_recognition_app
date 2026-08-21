"""6. Live Demo - the model predicts on stage.

Two paths: the committed sample gallery (the safe, rehearsed path) and a
free upload (the impressive path). The model is loaded once, cached, and
never trained here.
"""
import streamlit as st
from PIL import Image

from components import section_header
from model_utils import list_samples, load_bundle, predict, pretty


def _show_prediction(image, true_cls=None):
    col_img, col_res = st.columns([1, 1.4])
    with col_img:
        st.image(image, use_container_width=True)
    with col_res:
        results = predict(image, top_k=3)
        top_cls, top_p = results[0]

        if true_cls is not None:
            ok = top_cls == true_cls
            st.markdown(f"**True class:** {pretty(true_cls)}")
            (st.success if ok else st.error)(
                f"**Prediction: {pretty(top_cls)}** - {top_p:.1%} confidence"
            )
        else:
            st.info(f"**Prediction: {pretty(top_cls)}** - {top_p:.1%} confidence")

        st.caption("Top 3:")
        for cls, p in results:
            st.progress(p, text=f"{pretty(cls)} - {p:.1%}")


def render():
    section_header(
        "6. Live Demo",
        "A photograph goes in - crop and condition come out. "
        "DenseNet-121, loaded from the verified checkpoint, no training at runtime.",
    )

    # warm the cache on first visit so the prediction itself is instant
    load_bundle()

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
                "the model has never trained on them."
            )
            _show_prediction(Image.open(path), true_cls)

    with tab_upload:
        up = st.file_uploader("Leaf photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if up is not None:
            _show_prediction(Image.open(up))
        else:
            st.caption(
                "PlantVillage-style images (single leaf, plain background) "
                "play to the model's strengths - field photos are the known, "
                "openly discussed limitation (see Conclusion)."
            )
