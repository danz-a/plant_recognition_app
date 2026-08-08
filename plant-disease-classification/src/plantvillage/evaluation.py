"""The evaluation contract — written once, applied to every model.

Under ~36:1 class imbalance a model that ignores the rare classes can still
post a high accuracy, so the headline metric here is **macro-F1**, supported by
**per-class recall** and a **row-normalised confusion matrix**. Every model in
the project — dummy, logistic regression, random forest, CNN, transfer — is
scored by these same functions, which is what makes the comparison table honest.

No TensorFlow import: anything with a ``.predict`` method can be evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from plantvillage.utils import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Everything one model's evaluation produces, in one object."""

    name: str
    subset: str
    macro_f1: float
    accuracy: float
    per_class: pd.DataFrame
    confusion: np.ndarray
    class_names: Sequence[str] = field(repr=False, default=())
    y_true: np.ndarray = field(repr=False, default=None)
    y_pred: np.ndarray = field(repr=False, default=None)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EvaluationResult({self.name!r}, {self.subset}, "
            f"macro_F1={self.macro_f1:.3f}, accuracy={self.accuracy:.3f})"
        )

    @property
    def row(self) -> dict[str, Any]:
        return {"model": self.name, "macro_F1": round(self.macro_f1, 3), "accuracy": round(self.accuracy, 3)}

    def hardest(self, n: int = 5) -> pd.DataFrame:
        """The n classes with the lowest recall — where the imbalance bites."""
        return self.per_class.head(n)

    def easiest(self, n: int = 5) -> pd.DataFrame:
        return self.per_class.tail(n)

    def report(self, digits: int = 3) -> str:
        """Full precision / recall / F1 / support breakdown."""
        return classification_report(
            self.y_true,
            self.y_pred,
            target_names=list(self.class_names),
            digits=digits,
            zero_division=0,
        )

    def summary(self, n: int = 5) -> None:
        """Print the standard read: headline, worst classes, best classes."""
        print(f"{self.name}  [{self.subset}]:  macro-F1 = {self.macro_f1:.3f}   accuracy = {self.accuracy:.3f}\n")
        print(f"{n} hardest classes (lowest recall):")
        print(self.hardest(n).to_string(index=False))
        print(f"\n{n} easiest classes (highest recall):")
        print(self.easiest(n).to_string(index=False))

    def save(self, directory: Path, slug: str | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        slug = slug or self.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        self.per_class.to_csv(directory / f"{slug}_per_class_recall.csv", index=False)
        pd.DataFrame([self.row]).to_csv(directory / f"{slug}_metrics.csv", index=False)
        np.save(directory / f"{slug}_confusion.npy", self.confusion)
        logger.info("saved evaluation artifacts for %r to %s", self.name, directory)
        return directory


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    name: str,
    subset: str = "validation",
) -> EvaluationResult:
    """Compute the full evaluation contract from labels and predictions."""
    n_classes = len(class_names)
    per_class = pd.DataFrame(
        {
            "class": list(class_names),
            "recall": recall_score(
                y_true, y_pred, average=None, labels=range(n_classes), zero_division=0
            ),
            f"{subset}_images": np.bincount(y_true, minlength=n_classes),
        }
    ).sort_values("recall")

    return EvaluationResult(
        name=name,
        subset=subset,
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        per_class=per_class,
        confusion=confusion_matrix(y_true, y_pred, labels=range(n_classes)),
        class_names=list(class_names),
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
    )


def evaluate_model(
    model: Any,
    dataset: Any,
    y_true: np.ndarray,
    class_names: Sequence[str],
    name: str,
    subset: str = "validation",
) -> EvaluationResult:
    """Predict once, then evaluate.

    ``dataset`` must be unshuffled so the prediction order lines up with
    ``y_true`` — :func:`plantvillage.datasets.make_dataset` guarantees this for
    ``training=False``.
    """
    probabilities = model.predict(dataset, verbose=0)
    y_pred = np.asarray(probabilities).argmax(axis=1)
    return evaluate_predictions(y_true, y_pred, class_names, name, subset)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_confusion_matrix(result: EvaluationResult, figsize: tuple[int, int] = (14, 12)):
    """Row-normalised confusion matrix — the diagonal reads as per-class recall."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    counts = result.confusion.astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    normalised = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    figure = plt.figure(figsize=figsize)
    sns.heatmap(
        normalised,
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=list(result.class_names),
        yticklabels=list(result.class_names),
        cbar_kws={"shrink": 0.7, "label": "fraction of true class"},
    )
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.xlabel("predicted")
    plt.ylabel("true (diagonal = recall)")
    plt.title(f"{result.name} — row-normalised confusion matrix ({result.subset} set)")
    plt.tight_layout()
    return figure


def plot_learning_curves(
    history: Mapping[str, Sequence[float]],
    title: str = "Learning curves",
    phase_boundary: int | None = None,
):
    """Train vs validation accuracy and loss.

    The gap between the two curves is the over-fitting read that does not
    require touching the sealed test set.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, (metric, label) in zip(axes, [("accuracy", "Accuracy"), ("loss", "Loss")]):
        axis.plot(history[metric], label="train")
        axis.plot(history[f"val_{metric}"], label="validation")
        if phase_boundary is not None:
            axis.axvline(phase_boundary - 0.5, ls="--", c="grey", lw=1)
        axis.set_title(label)
        axis.set_xlabel("epoch")
        axis.legend()
    suffix = " (dashed = fine-tuning starts)" if phase_boundary is not None else ""
    plt.suptitle(f"{title} — learning curves{suffix}")
    plt.tight_layout()
    return figure


# --------------------------------------------------------------------------- #
# Comparisons
# --------------------------------------------------------------------------- #


def comparison_table(results: Sequence[EvaluationResult]) -> pd.DataFrame:
    """All evaluated models side by side, sorted by macro-F1."""
    return (
        pd.DataFrame([result.row for result in results])
        .set_index("model")
        .sort_values("macro_F1")
    )


def per_class_delta(
    baseline: EvaluationResult | pd.DataFrame,
    candidate: EvaluationResult | pd.DataFrame,
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
) -> pd.DataFrame:
    """Per-class recall difference between two models.

    On an already-strong model the aggregate barely moves; the interesting
    changes show up on the rare, visually similar classes, which is exactly
    what this table surfaces.
    """
    left = baseline.per_class if isinstance(baseline, EvaluationResult) else baseline
    right = candidate.per_class if isinstance(candidate, EvaluationResult) else candidate

    support_column = next((c for c in right.columns if c.endswith("_images")), None)
    merged = right.rename(columns={"recall": f"recall_{candidate_name}"}).merge(
        left[["class", "recall"]].rename(columns={"recall": f"recall_{baseline_name}"}),
        on="class",
    )
    merged["delta"] = (
        merged[f"recall_{candidate_name}"] - merged[f"recall_{baseline_name}"]
    ).round(3)

    columns = ["class"] + ([support_column] if support_column else []) + [
        f"recall_{baseline_name}",
        f"recall_{candidate_name}",
        "delta",
    ]
    return merged[columns].sort_values("delta", ascending=False)


def generalisation_gap(
    validation: EvaluationResult, test: EvaluationResult
) -> pd.DataFrame:
    """Validation vs test, side by side — the over-fitting check.

    A small gap says the model is stable *within* the PlantVillage lab
    distribution (plain backgrounds, even lighting). It does not say the model
    holds up on field photos — that needs a real-world test set.
    """
    table = pd.DataFrame(
        {
            "macro_F1": [validation.macro_f1, test.macro_f1],
            "accuracy": [validation.accuracy, test.accuracy],
        },
        index=["validation", "test"],
    ).round(3)
    table.loc["gap"] = (table.loc["validation"] - table.loc["test"]).round(3)
    return table
