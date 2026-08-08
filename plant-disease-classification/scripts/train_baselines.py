"""Classical baselines — the floors any real model must clear.

These are documentation floors, not rivals:

* ``dummy``   — no-skill reference. Shows accuracy staying non-trivial while
  macro-F1 collapses, which is the imbalance trap in one line.
* ``logreg``  — a *linear* model on the six global features.
* ``rf``      — a *non-linear* model on the same features (``--features global6``)
  or on the ~160-column histogram/texture descriptor (``--features rich``).

Running the linear and the non-linear model on identical inputs is what
separates "the model was too weak" from "the features lack the signal".

    python scripts/train_baselines.py --models dummy logreg rf
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from plantvillage import DataConfig, Paths
from plantvillage import data as pv_data
from plantvillage import evaluation, features
from plantvillage.utils import get_logger, save_json, set_seed

logger = get_logger("baselines")
RUN_NAME = "baselines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=None)
    parser.add_argument("--models", nargs="+", default=["dummy", "logreg", "rf"],
                        choices=["dummy", "logreg", "rf"])
    parser.add_argument("--features", default="global6", choices=["global6", "rich"],
                        help="feature set for the random forest")
    parser.add_argument("--train-per-class", type=int, default=300,
                        help="cap training images per class (0 = use the full split)")
    parser.add_argument("--n-estimators", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()

    paths = Paths.default(args.base)
    run_dir = paths.run(RUN_NAME)
    color_dir = pv_data.find_color_dir(paths.raw)
    class_names = pv_data.list_class_names(color_dir)
    name_to_index = {name: index for index, name in enumerate(class_names)}
    n_classes = len(class_names)

    split = pv_data.load_or_create_split(color_dir, class_names, paths.splits, DataConfig())

    # Training rows: capped by default so the floors fit in minutes, not hours.
    # Validation is always scored in full.
    if args.train_per_class:
        train_paths, train_labels = pv_data.subsample_per_class(
            split.train_paths, split.train_labels, args.train_per_class
        )
    else:
        train_paths, train_labels = split.train_paths, split.train_labels

    y_train = np.array([name_to_index[label] for label in train_labels])
    y_val = split.indices("val", name_to_index)

    results: list[evaluation.EvaluationResult] = []

    # ---- Floor A: no-skill dummy ------------------------------------------ #
    if "dummy" in args.models:
        placeholder_train = np.zeros((len(y_train), 1))
        placeholder_val = np.zeros((len(y_val), 1))
        for strategy in ("most_frequent", "stratified"):
            dummy = DummyClassifier(strategy=strategy, random_state=42).fit(placeholder_train, y_train)
            predictions = dummy.predict(placeholder_val)
            result = evaluation.evaluate_predictions(
                y_val, predictions, class_names, f"Dummy ({strategy})"
            )
            logger.info("%r", result)
            results.append(result)

    # ---- Floor B / C: models on the six global features ------------------- #
    if {"logreg", "rf"} & set(args.models):
        train_table = features.global_feature_table(
            train_paths, train_labels, cache=run_dir / "global6_train.csv"
        )
        val_table = features.global_feature_table(
            split.val_paths, split.val_labels, cache=run_dir / "global6_val.csv"
        )
        columns = list(features.GLOBAL_FEATURES)
        X_train = train_table[columns].to_numpy()
        X_val = val_table[columns].to_numpy()
        y_train_tab = train_table["label"].map(name_to_index).to_numpy()
        y_val_tab = val_table["label"].map(name_to_index).to_numpy()

        if "logreg" in args.models:
            # Scaling matters for a linear model; it is irrelevant for trees.
            scaler = StandardScaler().fit(X_train)
            logreg = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
            logreg.fit(scaler.transform(X_train), y_train_tab)
            result = evaluation.evaluate_predictions(
                y_val_tab,
                logreg.predict(scaler.transform(X_val)),
                class_names,
                "LogReg (6 global features)",
            )
            logger.info("%r", result)
            results.append(result)

        if "rf" in args.models and args.features == "global6":
            forest = RandomForestClassifier(
                n_estimators=args.n_estimators, class_weight="balanced", n_jobs=-1, random_state=42
            )
            forest.fit(X_train, y_train_tab)
            result = evaluation.evaluate_predictions(
                y_val_tab, forest.predict(X_val), class_names, "Random Forest (6 global features)"
            )
            logger.info("%r", result)
            results.append(result)

            importance = pd.Series(forest.feature_importances_, index=columns).sort_values(ascending=False)
            importance.to_csv(run_dir / "rf_global6_importance.csv", header=["importance"])
            print("\nFeature importance (sums to 1):")
            print(importance.round(3).to_string())

    # ---- Random forest on the rich descriptor ----------------------------- #
    if "rf" in args.models and args.features == "rich":
        spec = features.rich_feature_spec()
        logger.info("rich descriptor: %d columns %s", len(spec), spec.group_sizes)
        X_train, labels_train = features.rich_feature_matrix(
            train_paths, train_labels, cache=run_dir / "rich_train.npz"
        )
        X_val, labels_val = features.rich_feature_matrix(
            split.val_paths, split.val_labels, cache=run_dir / "rich_val.npz"
        )
        forest = RandomForestClassifier(
            n_estimators=args.n_estimators, class_weight="balanced", n_jobs=-1, random_state=42
        )
        forest.fit(X_train, np.array([name_to_index[label] for label in labels_train]))
        result = evaluation.evaluate_predictions(
            np.array([name_to_index[label] for label in labels_val]),
            forest.predict(X_val),
            class_names,
            "Random Forest (histograms + GLCM)",
        )
        logger.info("%r", result)
        results.append(result)

        by_group = (
            pd.Series(forest.feature_importances_, index=list(spec.groups))
            .groupby(level=0)
            .sum()
            .sort_values(ascending=False)
        )
        by_group.to_csv(run_dir / "rf_rich_group_importance.csv", header=["importance"])
        print("\nImportance by feature group (sums to 1):")
        print(by_group.round(3).to_string())

    # ---- Report ----------------------------------------------------------- #
    table = evaluation.comparison_table(results)
    print("\nValidation-set comparison (sorted by macro-F1):\n")
    print(table.to_string())
    table.to_csv(run_dir / "baseline_results.csv")
    for result in results:
        result.save(run_dir)
    save_json({"n_classes": n_classes, "n_train_fit": int(len(train_paths))}, run_dir / "run_config.json")
    logger.info("artifacts written to %s", run_dir)


if __name__ == "__main__":
    main()
