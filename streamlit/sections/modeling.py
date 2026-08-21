"""4. Modeling - eleven models, one pre-registered rule, one sealed test.

Content follows the defence script (Christoph's part). Figures load from
streamlit/assets/; if one is missing the slot shows a warning instead of
crashing the app.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from components import section_header

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def fig(name, caption=None):
    p = ASSETS / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.warning(f"Figure missing: assets/{name}")


def render():
    section_header(
        "4. Modeling",
        "Eleven models on one frozen split - macro-F1 decides, "
        "by a rule written down before the results existed.",
    )

    # ---- The measure and the rules --------------------------------------
    st.subheader("The measure and the rules")
    st.markdown(
        "The metric is **macro-F1** - the unweighted average of the 38 per-class "
        "F1 scores, so a rare disease counts as much as Tomato. The dummy model "
        "makes the case: always predicting the largest class gives **10.1 % "
        "accuracy** - a respectable-looking number for a model with no ability "
        "whatsoever. Its macro-F1 is **0.005**."
    )
    st.markdown(
        "Four rules, fixed in advance: every model on the **identical split** · "
        "**macro-F1 decides**, pre-registered · **validation** drives every "
        "choice · the **test set stays sealed** - opened once, at the end, "
        "for one model."
    )

    # ---- The ladder ------------------------------------------------------
    st.subheader("The ladder - eleven models, each answering a question")
    ladder = pd.DataFrame(
        {
            "Model": [
                "Dummy (most frequent)", "Dummy (stratified)",
                "Logistic regression - 6 global features",
                "Random Forest - 6 global features",
                "Random Forest - ~160 features (histograms + texture)",
                "CNN from scratch (4 blocks, 264k params)",
                "Scratch CNN + Bayesian tuning (15 trials)",
                "MobileNetV2 (transfer)", "ResNet50 (transfer)",
                "EfficientNetB0 (transfer)",
                "DenseNet-121 (two-phase fine-tune)",
            ],
            "Val macro-F1": [
                0.005, 0.026, 0.274, 0.382, 0.906,
                0.973, 0.970, 0.919, 0.978, 0.982, 0.9918,
            ],
        }
    )
    st.dataframe(ladder, hide_index=True, use_container_width=True)
    st.markdown(
        "The step worth remembering: the **same** Random Forest gains **+0.52** "
        "from richer features (0.382 → 0.906) - representation mattered about "
        "**five times more than the algorithm**. That is the most transferable "
        "thing we learned."
    )
    fig("densenet121_learning_curves.png",
        "DenseNet-121 two-phase fine-tuning: frozen-backbone head training, "
        "then unfreezing with BatchNorm kept frozen.")

    # ---- Choosing the final model ---------------------------------------
    st.subheader("Choosing the final model - and how thin the win is")
    st.markdown(
        "DenseNet-121 wins on the pre-registered rule, and the margin over the "
        "scratch CNN is close to noise - while the scratch CNN is **27× "
        "lighter**. If the goal were only this benchmark, the smaller model "
        "would be defensible. The goal is a tool for **field photographs**, "
        "which arrive in hundreds, not tens of thousands: DenseNet carries "
        "seven million parameters of general visual structure from 1.4 M "
        "natural images; the scratch CNN knows only detached leaves on grey "
        "card."
    )
    st.markdown(
        "The second argument is **where the errors went**: validation errors "
        "cut by **3.4×**, and in the tail - minimum per-class recall "
        "**0.9481**, only one class of 38 below 0.95, twenty of 38 recalled "
        "perfectly. Exactly what macro-F1 was chosen to reward."
    )

    # ---- The sealed evaluation ------------------------------------------
    st.subheader("The sealed evaluation - opened once")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test macro-F1", "0.9919", help="Bootstrap 95% CI: 0.9896-0.9941")
    c2.metric("Test accuracy", "0.9930")
    c3.metric("Errors", "57 / 8,146", help="One in 143 images")
    c4.metric("Val → Test gap", "+0.0001",
              help="0.9918 validation vs 0.9919 test")
    st.markdown(
        "The figure to point to is the **gap**: validation 0.9918, test 0.9919. "
        "No measurable over-fitting - and because the test set was never "
        "consulted during development, that statement carries information. "
        "The checkpoint is identified by **MD5 hash**, so the result is tied "
        "to a file, not to a description of one."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig("final_test_confusion_matrix.png", "Test confusion matrix - 38 classes.")
    with col_b:
        fig("final_test_per_class_val_vs_test.png",
            "Per-class recall, validation vs test.")

    # ---- Where the errors are -------------------------------------------
    st.subheader("Where the 57 errors are")
    st.markdown(
        "They fall into 31 pairs; ten pairs carry 63 % of them. Three patterns: "
        "**fine-grained confusion within a crop** (spider mites read as Target "
        "Spot - real diagnostic cost, treatments differ), visually similar "
        "conditions across crops, and singletons. What is **absent** matters "
        "too: none of the five largest pairs is a diseased leaf called healthy "
        "- the costliest failure mode. That absence is a direct consequence of "
        "optimising macro-F1."
    )
    fig("final_test_confusion_zoom.png",
        "Zoom on the most confused class pairs of the sealed test run.")
