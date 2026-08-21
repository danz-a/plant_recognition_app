import streamlit as st

from config import PRESENTATION_TITLE, SECTION_ORDER
from theme import apply_theme
from components import render_sidebar
from sections import (
    problem,
    exploration,
    preprocessing,
    modeling,
    interpretability,
    demo,
    conclusion,
)

st.set_page_config(
    page_title=PRESENTATION_TITLE,
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

sections = {
    "1. Business Problem / Introduction": problem.render,
    "2. Data Exploration": exploration.render,
    "3. Preprocessing": preprocessing.render,
    "4. Modeling": modeling.render,
    "5. Interpretability": interpretability.render,
    "6. Live Demo": demo.render,
    "7. Conclusion & Outlook": conclusion.render,
}

selected_section = render_sidebar(SECTION_ORDER)

st.title(PRESENTATION_TITLE)
st.divider()

sections[selected_section]()
