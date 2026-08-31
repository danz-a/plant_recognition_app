import streamlit as st


def section_header(title, description=None):
    st.header(title)

    if description:
        st.markdown(
            f'<p class="muted">{description}</p>',
            unsafe_allow_html=True,
        )
