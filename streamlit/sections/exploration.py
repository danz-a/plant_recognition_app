import streamlit as st
from components import section_header, placeholder


def render():
    section_header(
        "2. Data Exploration",
        "Placeholder section for dataset analysis, visualizations, and detected issues.",
    )

    st.subheader("Dataset Overview")

    placeholder(
        "Dataset Information",
        "Add information about the source, number of images, classes, and relevant features.",
    )

    st.subheader("Exploratory Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        placeholder(
            "Graph 1",
            "Insert graph here.",
        )

    with col2:
        placeholder(
            "Graph 2",
            "Insert graph here.",
        )

    st.subheader("Difficulties and Biases")

    placeholder(
        "Detected Issues",
        "Document class imbalance, image quality problems, dataset bias, duplicates, or other findings.",
    )
