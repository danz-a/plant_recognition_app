"""Evaluate a saved model on validation and, once, on the held-out test set.

The test set stays sealed through model selection; this script is the single
place it is opened. The read is simple: **test close to validation means the
model generalises; a large drop is over-fitting.**

One caveat worth stating in any report: validation and test come from the same
PlantVillage lab conditions, so a small gap confirms stability *within* that
distribution — not that the model survives field photos.

    python scripts/evaluate.py --run cnn_baseline
    python scripts/evaluate.py --model-path ~/plant_recognition/artifacts/cnn_tuned/model.keras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from plantvillage import DataConfig, Paths
from plantvillage import data as pv_data
from plantvillage import datasets, evaluation
from plantvillage.utils import get_logger, save_json, set_seed

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=None)
    parser.add_argument("--run", default="cnn_baseline", help="artifact folder holding model.keras")
    parser.add_argument("--model-path", default=None, help="explicit path to a .keras file")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--name", default=None, help="label used in the report")
    parser.add_argument("--skip-test", action="store_true", help="score validation only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()

    paths = Paths.default(args.base)
    run_dir = paths.run(args.run)
    model_path = Path(args.model_path) if args.model_path else run_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"no model at {model_path}. Train one first, or pass --model-path. "
            "(Colab wipes local storage between sessions — keep a copy on Drive.)"
        )

    data_config = DataConfig(img_size=args.img_size, batch_size=args.batch_size)
    color_dir = pv_data.find_color_dir(paths.raw)
    class_names = pv_data.list_class_names(color_dir)
    name_to_index = {name: index for index, name in enumerate(class_names)}

    split = pv_data.load_or_create_split(color_dir, class_names, paths.splits, data_config)
    subsets = ("val",) if args.skip_test else ("val", "test")
    pipelines = datasets.datasets_from_split(split, name_to_index, data_config, subsets=subsets)

    model = tf.keras.models.load_model(model_path)
    label = args.name or args.run

    validation = evaluation.evaluate_model(
        model, pipelines["val"], split.indices("val", name_to_index), class_names, label, "validation"
    )
    validation.summary()

    if args.skip_test:
        validation.save(run_dir, slug=f"{args.run}_validation")
        return

    test = evaluation.evaluate_model(
        model, pipelines["test"], split.indices("test", name_to_index), class_names, label, "test"
    )
    print("\n" + test.report())

    gap = evaluation.generalisation_gap(validation, test)
    print("\nValidation vs test:\n")
    print(gap.to_string())

    gap.to_csv(run_dir / "generalisation_gap.csv")
    test.save(run_dir, slug=f"{args.run}_test")
    save_json(
        {
            "model": str(model_path),
            "val_macro_F1": round(validation.macro_f1, 4),
            "test_macro_F1": round(test.macro_f1, 4),
            "gap_macro_F1": round(validation.macro_f1 - test.macro_f1, 4),
            "n_test": int(len(split.test_paths)),
        },
        run_dir / "test_evaluation.json",
    )
    logger.info("artifacts written to %s", run_dir)


if __name__ == "__main__":
    main()
