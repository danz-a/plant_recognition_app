"""3. Preprocessing - Scheima's part.

Content drafted from the team defence script; Scheima edits freely
(she is writing her own spoken script - this on-screen content should
be checked against it before the mock defence).
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
        "3. Preprocessing",
        "The audit showed there was nothing to repair. The job was not to "
        "fix the data - it was to organise it, correctly and without "
        "leakage, for a convolutional network.",
    )

    st.markdown(
        "No missing values, no broken files, constant geometry, three-channel "
        "RGB throughout. Every step below follows from one of the exploration "
        "findings."
    )

    # ---- The split -------------------------------------------------------
    st.subheader("The split - the most consequential step")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training", "~38,000", help="70 %")
    c2.metric("Validation", "8,146", help="15 %")
    c3.metric("Test (sealed)", "8,146", help="15 %")
    st.markdown(
        "Two failure modes had to be excluded: an **unstratified** split can "
        "under-represent rare classes at 36:1, and **leakage** between parts "
        "inflates every number downstream. So we split at **file level, "
        "before a single image is loaded**: every path and label enumerated, "
        "`train_test_split` applied twice with stratification. Three sets, "
        "**disjoint by construction**."
    )
    st.markdown(
        "We verified rather than assumed: assertions that all 38 classes "
        "appear in each set and that the sets are mutually disjoint. The "
        "split was written to CSV and **distributed as files** - every "
        "notebook prints its fingerprint at startup and halts on mismatch. "
        "That guard earned itself: it caught an early run that had quietly "
        "built a split of its own. The run was discarded and repeated."
    )
    st.info("Split fingerprint: `SPLIT_ID = 9e33ec57c1ec` - verifiable from "
            "the three CSVs in the repository.")

    # ---- The pipeline ----------------------------------------------------
    st.subheader("The pipeline - every step from a finding")
    st.markdown(
        "**Resize to 128 × 128** - recorded as a compute trade-off, not a "
        "data-driven necessity: the disease signal is local, so reducing "
        "resolution discards some of the very detail that carries it. "
        "Flagged for revisiting; it reappears in the error analysis."
    )
    st.markdown(
        "**Normalisation as a model layer.** 0-255 → 0-1 is a `Rescaling` "
        "layer at the model's input, not a change to the stored data. "
        "Training, validation and inference are guaranteed identical scaling "
        "- and the Streamlit app you are looking at can hand a raw "
        "photograph straight to the model."
    )
    st.markdown(
        "**Augmentation as layers** - horizontal/vertical flips, small "
        "rotations, small zooms. Keras augmentation layers are active only "
        "during training, so they can never touch validation or test."
    )
    st.markdown(
        "**Class weights** for the imbalance, computed on the training split, "
        "weighting the loss inversely to class frequency. Physical "
        "oversampling was rejected: it inflates the dataset, runs on CPU, "
        "and adds no signal."
    )
    st.markdown(
        "**No feature engineering, no PCA** - the exploration already showed "
        "the hand-built features are redundant, and reducing the pixel space "
        "would discard exactly the local structure the network needs."
    )

    st.subheader("What comes out")
    st.markdown(
        "Three batched datasets of 128 × 128 × 3 images across 38 classes, "
        "plus the layers and class weights the model applies. Christoph "
        "takes it from here."
    )
