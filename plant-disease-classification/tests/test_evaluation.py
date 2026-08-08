"""Tests for the evaluation contract.

The headline claim of the project is that accuracy is misleading under
imbalance while macro-F1 is not — so that is what these tests pin down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plantvillage.evaluation import (
    comparison_table,
    evaluate_predictions,
    generalisation_gap,
    per_class_delta,
)

CLASSES = ["a", "b", "c"]


def test_perfect_predictions_score_one():
    y = np.array([0, 1, 2, 0, 1, 2])
    result = evaluate_predictions(y, y, CLASSES, "perfect")
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.accuracy == pytest.approx(1.0)
    assert (result.per_class["recall"] == 1.0).all()


def test_majority_classifier_has_high_accuracy_but_low_macro_f1():
    # 90 of class 0, 5 each of classes 1 and 2 — the imbalance trap in miniature.
    y_true = np.array([0] * 90 + [1] * 5 + [2] * 5)
    y_pred = np.zeros_like(y_true)
    result = evaluate_predictions(y_true, y_pred, CLASSES, "majority")

    assert result.accuracy == pytest.approx(0.90)
    assert result.macro_f1 < 0.35  # the number that actually reveals the failure


def test_per_class_table_is_sorted_worst_first():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 0, 0])
    result = evaluate_predictions(y_true, y_pred, CLASSES, "uneven")
    recalls = result.per_class["recall"].tolist()
    assert recalls == sorted(recalls)
    assert result.per_class.iloc[0]["class"] == "c"


def test_confusion_matrix_is_square_and_complete():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 1])
    result = evaluate_predictions(y_true, y_pred, CLASSES, "small")
    assert result.confusion.shape == (3, 3)
    assert result.confusion.sum() == len(y_true)


def test_comparison_table_sorts_by_macro_f1():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    good = evaluate_predictions(y_true, y_true, CLASSES, "good")
    bad = evaluate_predictions(y_true, np.zeros_like(y_true), CLASSES, "bad")
    table = comparison_table([good, bad])
    assert list(table.index) == ["bad", "good"]


def test_per_class_delta_reports_signed_change():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    baseline = evaluate_predictions(y_true, np.array([0, 1, 1, 1, 2, 2]), CLASSES, "baseline")
    candidate = evaluate_predictions(y_true, y_true, CLASSES, "candidate")
    delta = per_class_delta(baseline, candidate)
    assert delta.loc[delta["class"] == "a", "delta"].item() == pytest.approx(0.5)
    assert delta["delta"].max() >= 0


def test_generalisation_gap_layout():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    validation = evaluate_predictions(y_true, y_true, CLASSES, "m", "validation")
    test = evaluate_predictions(y_true, np.array([0, 1, 2, 0, 1, 0]), CLASSES, "m", "test")
    gap = generalisation_gap(validation, test)
    assert list(gap.index) == ["validation", "test", "gap"]
    assert gap.loc["gap", "macro_F1"] > 0


def test_per_class_delta_accepts_a_dataframe():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    candidate = evaluate_predictions(y_true, y_true, CLASSES, "candidate")
    stored = pd.DataFrame({"class": CLASSES, "recall": [0.5, 0.5, 0.5]})
    delta = per_class_delta(stored, candidate)
    assert (delta["delta"] == 0.5).all()
