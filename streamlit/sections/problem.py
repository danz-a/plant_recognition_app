import streamlit as st
from components import section_header, placeholder


def render():
    section_header(
        "1. Business Problem / Introduction",
        "Placeholder section for the business problem and project motivation.",
    )

    st.subheader("Business Problem")

    placeholder(
        "Problem Statement",
        "Describe the practical problem that the project is intended to solve.",
    )

    placeholder(
        "Project Goal",
        "Define the desired outcome and explain why the problem matters.",
    )

    placeholder(
        "Potential Business Impact",
        "Explanation of how a successful solution could create value.",
    )
