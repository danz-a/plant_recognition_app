"""1. Business Problem / Introduction - Adrian's part.

Content drafted from the team defence script; Adrian edits freely.
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
        "1. Business Problem / Introduction",
        "A leaf-disease classifier: the problem it addresses, the data behind "
        "it - and, just as importantly, what it does not yet do.",
    )

    st.markdown(
        "**Team:** Adrian · Alexander · Christoph · Scheima — "
        "**Mentor:** Habiba El Husseiny"
    )
    st.markdown(
        "One framing point up front: the brief splits into two independent "
        "classifiers - species and disease - which meet at the application "
        "layer, where one photograph is shown to both and the answers are "
        "combined. Today we present the **disease half**, trained on "
        "PlantVillage."
    )

    # ---- Business problem ------------------------------------------------
    st.subheader("The business problem")
    st.markdown(
        "A plant disease caught in its **first week is a treatment**; the same "
        "disease a month later is a **lost field**. What stands between the "
        "two is diagnosis - and expert plant pathology is scarce, slow and "
        "expensive, often least available where the crops matter most."
    )
    st.markdown(
        "The target application: a grower photographs a leaf with an ordinary "
        "phone and receives, within seconds, the crop and its likely disease "
        "- in the spirit of PlantNet, but for disease. The value is a "
        "diagnosis cheaper and faster than an expert visit, delivered early "
        "enough to change what the grower actually does."
    )
    st.markdown(
        "Technically this is **fine-grained image classification** - and "
        "fine-grained is the whole difficulty. Not cat vs car: Tomato Early "
        "blight vs Tomato Late blight, two conditions on the same crop, where "
        "the distinguishing signal is a few square centimetres of texture."
    )

    # ---- The data --------------------------------------------------------
    st.subheader("The data")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images", "54,305", help="Colour photographs, 256x256 RGB")
    c2.metric("Classes", "38", help="Crop x condition, labelled by folder")
    c3.metric("Crops", "14")
    c4.metric("Conditions", "21", help="Healthy + 20 named diseases")
    st.markdown(
        "The dataset is **PlantVillage**, taken from Kaggle raw and "
        "unmodified. One point we are precise about: **twelve of the 38 "
        "classes are healthy tissue - this is a 38-class problem, not 38 "
        "diseases.** The archive also ships greyscale and background-removed "
        "copies of every photograph; we verified they are the same pictures "
        "in different dress, modelled on the colour images, and the segmented "
        "copies return later - not as training data but as a **measuring "
        "instrument** (section 5)."
    )
    st.markdown(
        "One deliberate choice: the raw archive rather than a pre-split "
        "repackaging. Every transformation downstream - split, augmentation, "
        "balancing - is one **we control and can account for**. That is what "
        "makes the pipeline reproducible."
    )
    fig("d1_sample_leaves_healthy_vs_diseased.jpg",
        "Healthy vs diseased sample leaves - the task at a glance.")
