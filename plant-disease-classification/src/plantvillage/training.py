"""Training loops and callbacks — one recipe, reused by every model.

Keeping the callbacks identical across the baseline, the tuned model and the
transfer models is what makes their validation scores comparable at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import tensorflow as tf
from tensorflow.keras import Model, callbacks as keras_callbacks

from plantvillage.config import TrainConfig
from plantvillage.utils import get_logger, timer

logger = get_logger(__name__)

History = dict[str, list[float]]


def default_callbacks(
    config: TrainConfig | None = None,
    checkpoint_path: Path | None = None,
) -> list[keras_callbacks.Callback]:
    """Early stopping on ``val_loss`` with best-weight restore, plus LR decay.

    We monitor loss but *judge* on macro-F1: under 36:1 imbalance accuracy is
    unreliable, and computing macro-F1 every epoch would need a second full
    prediction pass.
    """
    config = config or TrainConfig()
    result: list[keras_callbacks.Callback] = [
        keras_callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_lr,
        ),
    ]
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        result.append(
            keras_callbacks.ModelCheckpoint(
                str(checkpoint_path), monitor="val_loss", save_best_only=True
            )
        )
    return result


def fit(
    model: Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    epochs: int,
    class_weight: Mapping[int, float] | None = None,
    callbacks: Sequence[keras_callbacks.Callback] | None = None,
    label: str = "training",
    verbose: int = 1,
) -> History:
    """Train and return the history as a plain dict (JSON-serialisable)."""
    with timer(label):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=dict(class_weight) if class_weight else None,
            callbacks=list(callbacks) if callbacks else None,
            verbose=verbose,
        )
    result = {key: [float(v) for v in values] for key, values in history.history.items()}
    logger.info(
        "%s: %d epochs | best val_accuracy=%.3f",
        label,
        len(result["loss"]),
        max(result.get("val_accuracy", [float("nan")])),
    )
    return result


def train_transfer_model(
    model: Model,
    base_model: Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    class_weight: Mapping[int, float] | None = None,
    head_epochs: int = 5,
    finetune_epochs: int = 20,
    finetune_lr: float = 1e-5,
    last_n_layers: int | None = None,
    config: TrainConfig | None = None,
) -> tuple[History, int]:
    """The standard two-phase transfer recipe.

    Phase 1 trains the new head against a frozen backbone; phase 2 unfreezes and
    fine-tunes at a low learning rate. Doing phase 1 first matters: fine-tuning
    against a *random* head would push large, noisy gradients through the
    pre-trained weights on the very first batch.

    Returns the stitched history and the epoch index where phase 2 begins
    (used to draw the boundary on the learning curves).
    """
    from plantvillage.models import unfreeze_backbone

    head_history = fit(
        model,
        train_ds,
        val_ds,
        epochs=head_epochs,
        class_weight=class_weight,
        callbacks=[
            keras_callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
            )
        ],
        label="phase 1 — head warm-up (backbone frozen)",
    )

    unfreeze_backbone(
        model, base_model, learning_rate=finetune_lr, last_n_layers=last_n_layers
    )
    trainable = sum(1 for layer in base_model.layers if layer.trainable)
    logger.info("phase 2: %d/%d backbone layers trainable", trainable, len(base_model.layers))

    finetune_history = fit(
        model,
        train_ds,
        val_ds,
        epochs=finetune_epochs,
        class_weight=class_weight,
        callbacks=default_callbacks(config),
        label="phase 2 — fine-tuning",
    )

    stitched = {
        key: head_history[key] + finetune_history.get(key, [])
        for key in head_history
    }
    return stitched, len(head_history["loss"])


def save_history(history: History, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))
    return path


def load_history(path: Path) -> History:
    return json.loads(Path(path).read_text())


def load_or_train(
    model_path: Path,
    train_fn,
    *,
    force_retrain: bool = False,
) -> tuple[Model, History | None]:
    """Load a saved model if it exists, otherwise train it via ``train_fn``.

    Colab wipes its storage between sessions, so this is what lets an
    evaluation notebook run end-to-end whether or not the model survived.
    ``train_fn`` must return ``(model, history)``.
    """
    model_path = Path(model_path)
    history_path = model_path.with_suffix(".history.json")

    if model_path.exists() and not force_retrain:
        logger.info("loading saved model from %s", model_path)
        model = tf.keras.models.load_model(model_path)
        history = load_history(history_path) if history_path.exists() else None
        return model, history

    logger.info("no model at %s — training from scratch", model_path)
    model, history = train_fn()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    if history:
        save_history(history, history_path)
    return model, history
