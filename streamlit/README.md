# Streamlit Presentation Project

This is a minimal Streamlit project designed for a four-person machine learning presentation.

## Project structure

```text
presentation_streamlit/
├── app.py
├── config.py
├── theme.py
├── components.py
├── requirements.txt
├── README.md
├── assets/
└── sections/
    ├── __init__.py
    ├── problem.py
    ├── exploration.py
    ├── preprocessing.py
    ├── modeling.py
    └── interpretability.py
```

## Division of work

The presentation is divided into five independent sections:

1. Business Problem / Introduction
2. Data Exploration
3. Preprocessing
4. Modeling
5. Interpretability

Each section has its own Python file in `sections/`.

This allows different team members to work independently without normally editing the same file.

## Central design

The visual design is controlled centrally in:

```text
theme.py
```

Shared UI components are controlled in:

```text
components.py
```

General project settings and navigation order are controlled in:

```text
config.py
```

A team member working on an individual section should normally only need to edit the corresponding file in `sections/`.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

The application will open in the browser.

## Adding graphs

Graphs can be added directly to the relevant section.

For example:

```python
st.line_chart(data)
```

or with Matplotlib:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)
```

Additional Python packages can be added to `requirements.txt` if required.

## Recommended Git workflow

Each team member can work primarily on their assigned section:

```text
Person 1 → sections/problem.py
Person 2 → sections/exploration.py
Person 3 → sections/preprocessing.py
Person 4 → sections/modeling.py
```

`interpretability.py` can be assigned to one of the four members or developed jointly.

Central design changes should be made in `theme.py`, ideally by agreement of the team to avoid merge conflicts.

## Important

The current content is intentionally only a placeholder. The project does not contain the actual dataset, models, or Grad-CAM implementation yet.
