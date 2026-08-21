import streamlit as st

# Central design configuration.
# Edit this file to change the visual appearance of the whole presentation.

PRIMARY_COLOR = "#2E7D32"
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#222222"
MUTED_COLOR = "#666666"


def apply_theme():
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
            }}

            h1, h2, h3 {{
                color: {PRIMARY_COLOR};
            }}

            .presentation-card {{
                padding: 1.25rem;
                border-radius: 0.75rem;
                border: 1px solid #DDDDDD;
                background-color: #FAFAFA;
                margin-bottom: 1rem;
            }}

            .muted {{
                color: {MUTED_COLOR};
            }}

            section[data-testid="stSidebar"] {{
                border-right: 1px solid #DDDDDD;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
