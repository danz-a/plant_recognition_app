"""7. Conclusion & Outlook - the result, its honest limits, and what's next.

Covers the two required defence elements: conclusion linked to the business
issue, and criticism & outlook.
"""
import streamlit as st

from components import section_header


def render():
    section_header(
        "7. Conclusion & Outlook",
        "What was achieved, what it means for the business problem, "
        "and what we would do with more time.",
    )

    # ---- Conclusion ------------------------------------------------------
    st.subheader("Conclusion - linked to the business issue")
    c1, c2, c3 = st.columns(3)
    c1.metric("Final model", "DenseNet-121")
    c2.metric("Sealed-test macro-F1", "0.9919",
              help="95% CI 0.9896-0.9941 · evaluated exactly once")
    c3.metric("Errors", "1 in 143",
              help="57 of 8,146 test images")
    st.markdown(
        "The business problem was scarce, slow expert diagnosis. On "
        "PlantVillage-style imagery this classifier answers in under a second "
        "at an error rate of one in 143 - with the rare classes protected "
        "(minimum per-class recall 0.9481) and **no dominant "
        "diseased-called-healthy failure mode**, the costliest error a "
        "grower-facing tool could make. The pipeline behind it is fully "
        "reproducible: one frozen split with a verifiable fingerprint, "
        "pre-registered decision rules, and a final checkpoint identified "
        "by MD5 hash."
    )

    # ---- Criticism -------------------------------------------------------
    st.subheader("Criticism - the honest limits")
    st.markdown(
        "**Lab imagery, not field imagery.** Every number here is measured on "
        "detached leaves against a grey card. We make no field-robustness "
        "claim - and our own interpretability analysis is the reason: the "
        "pre-registered hypothesis that the pretrained backbone relies less "
        "on the backdrop was **not supported**. "
        "**Single-condition crops.** Blueberry, Raspberry and Soybean appear "
        "only healthy, Orange and Squash only diseased - for those crops, "
        "healthy-vs-diseased is unlearnable from this data. "
        "**The concentration metric needs a mask.** Our attention measurement "
        "depends on segmentation outlines that field photos will not have."
    )

    # ---- Outlook ---------------------------------------------------------
    st.subheader("Outlook - with more time")
    st.markdown(
        "**First, field evaluation:** a few hundred labelled field photographs "
        "would tell us more than any further tuning on PlantVillage - and "
        "DenseNet's pretrained features are precisely the asset for adapting "
        "on data that scarce. "
        "**Second, validate the diffuseness proxy:** if spread-out attention "
        "predicts errors without a mask, Grad-CAM becomes a runtime "
        "confidence signal - flag diffuse predictions for human review "
        "instead of returning a silent wrong answer. "
        "**Third, deployment:** the inference model is 30 MB and CPU-friendly; "
        "on-device operation for offline field use is realistic."
    )

    st.divider()
    st.markdown(
        "*Thank you - we're happy to take questions. The complete pipeline, "
        "split fingerprints, sealed-test protocol and this app are in the "
        "repository.*"
    )
