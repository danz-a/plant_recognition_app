import streamlit as st

from config import PRESENTATION_TITLE
from theme import apply_theme
from sections import demo

st.set_page_config(
    page_title=PRESENTATION_TITLE,
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

st.title(PRESENTATION_TITLE)
st.divider()

demo.render()
