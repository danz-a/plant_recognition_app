import streamlit as st
from components import section_header, placeholder


def render():
    section_header(
        "3. Preprocessing",
        "Placeholder section for data cleaning and preparation.",
    )

    st.subheader("Data Cleaning")

    placeholder(
        "Cleaning Steps",
        "Describe data handling",
    )

    st.subheader("Data Transformation")

    placeholder(
        "Preprocessing Pipeline",
        "Describe transformations.",
    )

    st.subheader("Resolution of Identified Issues")

    placeholder(
        "Problem → Solution",
        "Explain how the issues identified during data exploration were addressed.",
    )
