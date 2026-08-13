"""7. Conclusion & Outlook.

Covers the two required defence elements the five original sections do not:
'conclusion linked to the business issue' and 'criticism and outlook'.
Owner: Christoph - replace the placeholders with the final text.
"""
from components import section_header, placeholder


def render():
    section_header(
        "7. Conclusion & Outlook",
        "What was achieved, what it means for the business problem, "
        "and what we would do with more time.",
    )

    placeholder(
        "Conclusion - linked to the business issue",
        "Headline result (DenseNet-121, macro-F1 0.9919 on the sealed test set) "
        "and what that level of accuracy would mean for the grower-facing "
        "diagnosis tool.",
    )

    placeholder(
        "Criticism - honest limits",
        "PlantVillage is lab-style imagery; field robustness is unproven. "
        "The Grad-CAM concentration analysis explicitly does NOT support a "
        "field-robustness claim - say so before the jury asks.",
    )

    placeholder(
        "Outlook - with more time",
        "Field-image evaluation, mask-free attention proxies (untested "
        "alternatives, framed as such), deployment considerations.",
    )
