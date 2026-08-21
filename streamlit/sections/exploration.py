"""2. Data Exploration - Alexander's part.

Content drafted from the team defence script; Alexander edits freely.
Five figures, five findings - each with a plain reading and a test.
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
        "2. Data Exploration",
        "One discipline throughout: every figure gets a plain reading and a "
        "statistical test. The picture tells you what to think; the test "
        "tells you whether you are entitled to think it.",
    )

    # ---- Finding 1 -------------------------------------------------------
    st.subheader("1 · Severe class imbalance")
    fig("d1_class_distribution.jpg",
        "Images per class - 38 classes, n = 54,305.")
    st.markdown(
        "Largest class: Orange citrus greening, **5,507** images. Smallest: "
        "healthy Potato, **152**. An imbalance of about **36 : 1**. The "
        "chi-square rejection (~41,874, p ≈ 0) was never in doubt at this "
        "sample size - the number that measures severity is the **ratio**, "
        "not the p-value. Consequence: plain accuracy would flatter a bad "
        "model, so we score on **macro-F1 and per-class recall** and correct "
        "the imbalance during training."
    )

    # ---- Finding 2 -------------------------------------------------------
    st.subheader("2 · Crop and disease are entangled")
    fig("d1_healthy_vs_diseased_per_crop.jpg",
        "Healthy vs diseased images per crop.")
    st.markdown(
        "Tomato alone is about a **third of the dataset**. Several crops "
        "appear under one condition only - Blueberry, Raspberry and Soybean "
        "entirely healthy; Orange and Squash entirely diseased "
        "(independence rejected: chi-square ≈ 28,005 on 13 df). For those "
        "crops, 'healthy or diseased?' **cannot be learned** - the data "
        "contains only one of the two answers. No augmentation fixes that; "
        "it is why the task is framed as **flat 38-class prediction** rather "
        "than crop-then-disease."
    )

    # ---- Finding 3 -------------------------------------------------------
    st.subheader("3 · The most important finding: disease is local")
    fig("d1_leaf_greenness_boxplot.jpg",
        "Green fraction G/(R+G+B), healthy vs diseased - crops with both "
        "conditions only.")
    st.markdown(
        "Can one global colour summary separate healthy from diseased? "
        "Medians **0.358 vs 0.354** - the boxes sit almost on top of one "
        "another. ANOVA calls it significant (F ≈ 94.8, p ≈ 10⁻²²) but the "
        "effect size is **η² = 0.011**: health status explains about **one "
        "percent** of greenness variation. Real and unimportant - "
        "significant only because the test runs on eight thousand leaves."
    )
    st.markdown(
        "The reason is the whole project in one sentence: **disease is "
        "local**. A diseased leaf is mostly healthy green tissue with a few "
        "spots, and averaging the whole leaf washes the symptom out. That is "
        "the empirical case for a convolutional network - made with numbers, "
        "not asserted."
    )

    # ---- Findings 4 + 5 --------------------------------------------------
    st.subheader("4 · No tabular shortcut")
    fig("d1_feature_correlation_heatmap.jpg",
        "Global feature correlations.")
    st.markdown(
        "Brightness vs green correlates at **0.94**, red vs green at "
        "**0.88** - four colour numbers measuring one underlying lightness "
        "dimension. There was no tabular shortcut available."
    )

    st.subheader("5 · We simply looked")
    st.markdown(
        "Healthy leaves are evenly green; diseased leaves carry **localised** "
        "damage. And every single leaf sits on a plain laboratory background "
        "- a shortcut a model could exploit instead of learning the disease. "
        "We come back to that twice more."
    )

    # ---- Carry-forward ---------------------------------------------------
    st.subheader("Four difficulties to carry forward")
    st.markdown(
        "**Severe class imbalance** · **crop-disease entanglement** · "
        "**Tomato dominance** · **the clean laboratory background.** "
        "Scheima explains what we did about all of it."
    )
