import streamlit as st


def render_sidebar(section_order):
    st.sidebar.title("Presentation")

    selected = st.sidebar.radio(
        "Select section",
        section_order,
    )

    st.sidebar.divider()
    st.sidebar.caption("Central navigation and design are managed separately.")

    return selected


def section_header(title, description=None):
    st.header(title)

    if description:
        st.markdown(
            f'<p class="muted">{description}</p>',
            unsafe_allow_html=True,
        )


def placeholder(title, text):
    st.markdown(
        f"""
        <div class="presentation-card">
            <h3>{title}</h3>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
