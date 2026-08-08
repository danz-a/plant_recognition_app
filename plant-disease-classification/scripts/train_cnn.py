"""Train the from-scratch baseline CNN.

Deliberately small and un-tuned: this is the reference every later improvement
is measured against, so it has to be honest rather than impressive.

    python scripts/train_cnn.py --epochs 20
    python scripts/train_cnn.py --smoke-test        # ~2-minute dry run
"""

from __future__ import annotations

import argparse

from plantvillage import DataConfig, Paths, TrainConfig
from plantvillage import data as pv_data
from plantvillage import datasets, evaluation, models, training
from plantvillage.config import RunSummary
from plantvillage.utils import describe_hardware, get_logger, set_seed

logger = get_logger("train_cnn")
RUN_NAME = "cnn_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=None)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="60 images/class, 2 epochs")
    parser.add_argument("--run-name", default=RUN_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    logger.info(describe_hardware())

    data_config = DataConfig(img_size=args.img_size, batch_size=args.batch_size)
    train_config = TrainConfig(
        epochs=2 if args.smoke_test else args.epochs,
        learning_rate=args.learning_rate,
        class_weights=not args.no_class_weights,
    )

    paths = Paths.default(args.base)
    run_dir = paths.run(args.run_name)
    color_dir = pv_data.find_color_dir(paths.raw)
    class_names = pv_data.list_class_names(color_dir)
    name_to_index = {name: index for index, name in enumerate(class_names)}

    split = pv_data.load_or_create_split(color_dir, class_names, paths.splits, data_config)
    if args.smoke_test:
        train_paths, train_labels = pv_data.subsample_per_class(split.train_paths, split.train_labels, 60)
        val_paths, val_labels = pv_data.subsample_per_class(split.val_paths, split.val_labels, 20)
        split = pv_data.Split(train_paths, train_labels, val_paths, val_labels,
                              split.test_paths, split.test_labels)
        logger.info("[smoke test] train=%d val=%d", len(train_paths), len(val_paths))

    pipelines = datasets.datasets_from_split(split, name_to_index, data_config)
    datasets.assert_raw_pixels(pipelines["train"])

    y_train = split.indices("train", name_to_index)
    y_val = split.indices("val", name_to_index)
    class_weight = (
        pv_data.balanced_class_weights(y_train, len(class_names))
        if train_config.class_weights
        else None
    )
    if class_weight:
        logger.info(
            "class weights: %.2f (most common) ... %.2f (rarest)",
            min(class_weight.values()),
            max(class_weight.values()),
        )

    model = models.build_baseline_cnn(
        n_classes=len(class_names),
        img_size=data_config.img_size,
        learning_rate=train_config.learning_rate,
    )
    model.summary()

    history = training.fit(
        model,
        pipelines["train"],
        pipelines["val"],
        epochs=train_config.epochs,
        class_weight=class_weight,
        callbacks=training.default_callbacks(train_config),
        label="baseline CNN",
    )

    result = evaluation.evaluate_model(
        model, pipelines["val"], y_val, class_names, "Baseline CNN (from scratch)"
    )
    result.summary()
    print("\n" + result.report())

    model.save(run_dir / "model.keras")
    training.save_history(history, run_dir / "model.history.json")
    result.save(run_dir, slug="cnn_baseline")
    evaluation.comparison_table([result]).to_csv(run_dir / "results.csv")
    RunSummary(
        name=args.run_name,
        data=data_config,
        train=train_config,
        extra={
            "epochs_run": len(history["loss"]),
            "macro_F1": round(result.macro_f1, 4),
            "accuracy": round(result.accuracy, 4),
            **{f"n_{k}": v for k, v in split.sizes.items()},
        },
    ).save(run_dir)
    logger.info("artifacts written to %s", run_dir)


if __name__ == "__main__":
    main()
