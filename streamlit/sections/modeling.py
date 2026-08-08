import streamlit as st
from components import section_header, placeholder


def render():
    section_header(
        "4. Modeling",
        "Placeholder section for comparing different machine learning approaches.",
    )

    st.subheader("Approaches")

    data = {
        "Approach": [
            "Baseline Model",
            "Classical Machine Learning",
            "Deep Learning",
            "Transfer Learning",
        ],
        "Description": [
            "Simple reference model.",
            "Traditional machine learning approach.",
            "Neural network trained on the dataset.",
            "Pre-trained model adapted to the task.",
        ],
        "Performance": [
            "TBD",
            "TBD",
            "TBD",
            "TBD",
        ],
    }

    st.table(data)

    st.subheader("Model Selection")

    placeholder(
        "Selected Model",
        "Explain which approach performed best",
    )

    placeholder(
        "Evaluation",
        "Add evaluation",
    )
