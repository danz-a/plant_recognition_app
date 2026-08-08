"""Bayesian hyper-parameter search for the baseline CNN.

Configurations are *ranked* on a stratified sub-sample of the training data —
searching all ~38k images per trial would cost hours for a coarse decision like
learning rate. The winner is then **retrained on the full training set**, so the
reported number is never the sub-sample's.

Everything else (split, resolution, batch size, class weights, callbacks) is
inherited unchanged, which is what makes "did tuning help?" a fair question.

    python scripts/tune_cnn.py --max-trials 15
"""

from __future__ import annotations

import argparse
import json

import keras_tuner as kt
import pandas as pd
import tensorflow as tf

from plantvillage import DataConfig, Paths, TrainConfig
from plantvillage import data as pv_data
from plantvillage import datasets, evaluation, models, training
from plantvillage.utils import describe_hardware, get_logger, set_seed

logger = get_logger("tune_cnn")
RUN_NAME = "cnn_tuned"
BASELINE_RUN = "cnn_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=None)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-trials", type=int, default=15)
    parser.add_argument("--search-epochs", type=int, default=10)
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument("--tune-per-class", type=int, default=250,
                        help="training images per class used for ranking (0 = full split)")
    parser.add_argument("--run-name", default=RUN_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    logger.info(describe_hardware())

    data_config = DataConfig(img_size=args.img_size, batch_size=args.batch_size)
    train_config = TrainConfig(epochs=args.final_epochs)

    paths = Paths.default(args.base)
    run_dir = paths.run(args.run_name)
    color_dir = pv_data.find_color_dir(paths.raw)
    class_names = pv_data.list_class_names(color_dir)
    name_to_index = {name: index for index, name in enumerate(class_names)}
    n_classes = len(class_names)

    split = pv_data.load_or_create_split(color_dir, class_names, paths.splits, data_config)
    pipelines = datasets.datasets_from_split(split, name_to_index, data_config)

    y_train = split.indices("train", name_to_index)
    y_val = split.indices("val", name_to_index)
    class_weight = pv_data.balanced_class_weights(y_train, n_classes)

    # ---- the search sub-sample (training data only — no leakage) ---------- #
    if args.tune_per_class:
        search_paths, search_labels = pv_data.subsample_per_class(
            split.train_paths, split.train_labels, args.tune_per_class
        )
    else:
        search_paths, search_labels = split.train_paths, split.train_labels
    y_search = [name_to_index[label] for label in search_labels]
    search_ds = datasets.make_dataset(search_paths, y_search, data_config, training=True)
    search_weights = pv_data.balanced_class_weights(y_search, n_classes)
    logger.info("search sub-sample: %d images (full train: %d)", len(search_paths), len(split.train_paths))

    # ---- Bayesian search -------------------------------------------------- #
    def build(hp):
        return models.build_baseline_cnn(
            n_classes=n_classes, img_size=data_config.img_size, **models.hyperparameter_space(hp)
        )

    tuner = kt.BayesianOptimization(
        build,
        objective="val_accuracy",  # a ranking signal only; the winner is judged on macro-F1
        max_trials=args.max_trials,
        seed=data_config.seed,
        directory=str(run_dir),
        project_name="bayesian_search",
        overwrite=False,  # an interrupted search resumes instead of restarting
    )
    tuner.search_space_summary()
    tuner.search(
        search_ds,
        validation_data=pipelines["val"],
        epochs=args.search_epochs,
        class_weight=search_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    logger.info("best hyper-parameters: %s", best_hp.values)
    (run_dir / "best_hyperparameters.json").write_text(json.dumps(best_hp.values, indent=2))

    trials = pd.DataFrame(
        [
            {"trial": trial.trial_id, "val_accuracy_subsample": round(trial.score, 4), **trial.hyperparameters.values}
            for trial in tuner.oracle.get_best_trials(min(5, args.max_trials))
        ]
    )
    print("\nTop trials:\n")
    print(trials.to_string(index=False))
    trials.to_csv(run_dir / "top_trials.csv", index=False)

    # ---- retrain the winner on the FULL training set ---------------------- #
    model = build(best_hp)
    history = training.fit(
        model,
        pipelines["train"],
        pipelines["val"],
        epochs=args.final_epochs,
        class_weight=class_weight,
        callbacks=training.default_callbacks(train_config),
        label="tuned CNN — full-data retrain",
    )

    result = evaluation.evaluate_model(
        model, pipelines["val"], y_val, class_names, "Tuned CNN (Step 3.2)"
    )
    result.summary()

    model.save(run_dir / "model.keras")
    training.save_history(history, run_dir / "model.history.json")
    result.save(run_dir, slug="cnn_tuned")

    # ---- comparison against the saved baseline ---------------------------- #
    baseline_per_class = paths.artifacts / BASELINE_RUN / "cnn_baseline_per_class_recall.csv"
    if baseline_per_class.exists():
        delta = evaluation.per_class_delta(
            pd.read_csv(baseline_per_class), result, "baseline", "tuned"
        )
        delta.to_csv(run_dir / "baseline_vs_tuned_per_class.csv", index=False)
        print("\nBiggest recall improvements (tuned - baseline):")
        print(delta.head(6).to_string(index=False))
        print("\nBiggest regressions:")
        print(delta.tail(6).to_string(index=False))
        improved = int((delta["delta"] > 0).sum())
        regressed = int((delta["delta"] < 0).sum())
        print(
            f"\nMean per-class recall change: {delta['delta'].mean():+.3f} | "
            f"improved {improved}, regressed {regressed}, unchanged {n_classes - improved - regressed}"
        )
    else:
        logger.warning("no baseline per-class recalls at %s — run train_cnn.py first", baseline_per_class)

    logger.info("artifacts written to %s", run_dir)


if __name__ == "__main__":
    main()
