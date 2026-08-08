"""Transfer learning — one script for every ImageNet backbone.

DenseNet-121 and ResNet-50 were separate notebooks with separate conventions;
here they are one code path and one ``--backbone`` flag, so their validation
scores are directly comparable.

Two phases: warm up the new head against a frozen backbone, then fine-tune at a
much lower learning rate.

    python scripts/train_transfer.py --backbone densenet121
    python scripts/train_transfer.py --backbone resnet50 --img-size 224 --last-n-layers 4
"""

from __future__ import annotations

import argparse

from plantvillage import DataConfig, Paths, TrainConfig
from plantvillage import data as pv_data
from plantvillage import datasets, evaluation, models, training
from plantvillage.config import RunSummary
from plantvillage.utils import describe_hardware, get_logger, set_seed

logger = get_logger("train_transfer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=None)
    parser.add_argument("--backbone", default="densenet121", choices=sorted(models.BACKBONES))
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--last-n-layers", type=int, default=None,
                        help="fine-tune only the last N backbone layers (default: all)")
    parser.add_argument("--hidden-units", type=int, nargs="*", default=[],
                        help="extra dense layers in the head, e.g. --hidden-units 1024 512")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    logger.info(describe_hardware())

    run_name = args.run_name or f"transfer_{args.backbone}"
    data_config = DataConfig(img_size=args.img_size, batch_size=args.batch_size)
    train_config = TrainConfig(epochs=args.finetune_epochs, learning_rate=args.head_lr)

    paths = Paths.default(args.base)
    run_dir = paths.run(run_name)
    color_dir = pv_data.find_color_dir(paths.raw)
    class_names = pv_data.list_class_names(color_dir)
    name_to_index = {name: index for index, name in enumerate(class_names)}

    split = pv_data.load_or_create_split(color_dir, class_names, paths.splits, data_config)
    if args.smoke_test:
        train_paths, train_labels = pv_data.subsample_per_class(split.train_paths, split.train_labels, 60)
        val_paths, val_labels = pv_data.subsample_per_class(split.val_paths, split.val_labels, 20)
        split = pv_data.Split(train_paths, train_labels, val_paths, val_labels,
                              split.test_paths, split.test_labels)

    pipelines = datasets.datasets_from_split(split, name_to_index, data_config)
    datasets.assert_raw_pixels(pipelines["train"])

    y_train = split.indices("train", name_to_index)
    y_val = split.indices("val", name_to_index)
    class_weight = pv_data.balanced_class_weights(y_train, len(class_names))

    model, base_model = models.build_transfer_model(
        backbone=args.backbone,
        n_classes=len(class_names),
        img_size=data_config.img_size,
        hidden_units=tuple(args.hidden_units),
        learning_rate=args.head_lr,
    )
    model.summary()

    history, phase_boundary = training.train_transfer_model(
        model,
        base_model,
        pipelines["train"],
        pipelines["val"],
        class_weight=class_weight,
        head_epochs=2 if args.smoke_test else args.head_epochs,
        finetune_epochs=2 if args.smoke_test else args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        last_n_layers=args.last_n_layers,
        config=train_config,
    )

    result = evaluation.evaluate_model(
        model, pipelines["val"], y_val, class_names, f"{args.backbone} (transfer)"
    )
    result.summary()
    print("\n" + result.report())

    model.save(run_dir / "model.keras")
    training.save_history(history, run_dir / "model.history.json")
    result.save(run_dir, slug=args.backbone)
    RunSummary(
        name=run_name,
        data=data_config,
        train=train_config,
        extra={
            "backbone": args.backbone,
            "head_epochs": args.head_epochs,
            "finetune_lr": args.finetune_lr,
            "phase_boundary": phase_boundary,
            "macro_F1": round(result.macro_f1, 4),
            "accuracy": round(result.accuracy, 4),
        },
    ).save(run_dir)
    logger.info("artifacts written to %s", run_dir)


if __name__ == "__main__":
    main()
