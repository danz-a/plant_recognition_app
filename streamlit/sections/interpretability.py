"""5. Interpretability - Grad-CAM turned from pictures into a measurement.

Includes the pre-registered hypothesis that was NOT supported (disclosed
proactively, per the report) and the correct-vs-wrong finding that was.
"""
from pathlib import Path

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
        "5. Interpretability",
        "A model right 99.3 % of the time still answers without a reason. "
        "Grad-CAM recovers something close to one - and we turned it into "
        "a measurement.",
    )

    # ---- What Grad-CAM shows --------------------------------------------
    st.subheader("From heat maps to a number")
    st.markdown(
        "Grad-CAM reads the last convolutional layer and marks how strongly "
        "each region pushed the prediction towards the chosen class - warm "
        "where the evidence came from. Heat maps are persuasive and easy to "
        "over-read, so we made them measurable: the segmented PlantVillage "
        "images give a **free leaf outline neither network ever saw**. "
        "**Concentration** = share of heat inside the leaf ÷ share of frame "
        "the leaf occupies. 1.0 means no preference; above 1.0, the model "
        "prefers the leaf."
    )
    fig("gradcam_paired_panel.png",
        "Same images, both models: scratch CNN vs DenseNet-121 attention.")

    # ---- The numbers -----------------------------------------------------
    st.subheader("What the numbers say")
    c1, c2, c3 = st.columns(3)
    c1.metric("Leaf share of frame", "47.4 %",
              help="Across the 7,967 validation images with a segmented counterpart")
    c2.metric("Scratch CNN concentration", "1.169",
              help="56.7 % of attention on the leaf")
    c3.metric("DenseNet-121 concentration", "1.134",
              help="53.3 % of attention on the leaf")
    st.markdown(
        "Both above 1.0 - neither model reads the backdrop in preference to "
        "the plant. But both leave a large share of attention on plain grey "
        "background carrying no disease information at all: a useful "
        "corrective to scores in the high nineties."
    )
    fig("attention_on_leaf.png",
        "Distribution of attention-on-leaf across the validation set.")

    # ---- The disclosure --------------------------------------------------
    st.subheader("A pre-registered hypothesis - not supported")
    st.markdown(
        "We had written down, in advance, that the pretrained backbone would "
        "depend *less* on the laboratory backdrop. **It does not.** DenseNet "
        "concentrates on the leaf slightly less (1.134 vs 1.169), and the "
        "difference survives a control for its coarser last-layer grid. We "
        "report it, and we make **no field-robustness claim** for the final "
        "model anywhere in the report."
    )

    # ---- The finding that survived --------------------------------------
    st.subheader("The finding that survived")
    c1, c2, c3 = st.columns(3)
    c1.metric("Concentration when correct", "1.135")
    c2.metric("Concentration when wrong", "1.009",
              help="No leaf preference at all")
    c3.metric("Difference", "0.126",
              help="95% CI 0.053-0.203 · Mann-Whitney p = 7×10⁻⁴")
    st.markdown(
        "A concentration of 1.009 is no leaf preference at all. When this "
        "model is wrong, its attention is spread **as though the leaf were "
        "not there** - misclassification coincides, measurably, with reading "
        "the wrong part of the frame."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        fig("attention_correct_vs_wrong.png",
            "Concentration split by prediction outcome.")
    with col_b:
        fig("gradcam_densenet_failures.png",
            "DenseNet errors: diffuse attention on misclassified images.")

    st.markdown(
        "**The honest caveat:** concentration needs the segmentation mask, and "
        "a field photograph does not come with one. What carries over is how "
        "*spread out* the heat map is - a proxy we would validate before "
        "trusting. If it holds, Grad-CAM turns from a presentation device "
        "into a runtime confidence signal: flag diffuse predictions for "
        "human review."
    )
