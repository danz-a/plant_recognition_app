"""Model architectures.

Two families, one convention: **every model normalises its own input**. The
first layers are always pre-processing + train-only augmentation, so a saved
model is a complete, self-contained predictor for a raw 0-255 image.
"""

from __future__ import annotations

from typing import Callable

import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers

from plantvillage.config import SEED

# --------------------------------------------------------------------------- #
# Backbone registry
# --------------------------------------------------------------------------- #

BACKBONES: dict[str, tuple[Callable[..., Model], Callable]] = {
    "densenet121": (
        tf.keras.applications.DenseNet121,
        tf.keras.applications.densenet.preprocess_input,
    ),
    "resnet50": (
        tf.keras.applications.ResNet50,
        tf.keras.applications.resnet.preprocess_input,
    ),
    "efficientnetb0": (
        tf.keras.applications.EfficientNetB0,
        tf.keras.applications.efficientnet.preprocess_input,
    ),
}


@tf.keras.utils.register_keras_serializable(package="plantvillage")
class BackbonePreprocessing(layers.Layer):
    """Applies the backbone's own ImageNet pre-processing, inside the model.

    Wrapping ``preprocess_input`` in a registered layer (rather than mapping it
    over the dataset) keeps the saved model self-contained *and* serialisable —
    the two properties a Lambda layer would cost us.
    """

    def __init__(self, backbone: str, **kwargs) -> None:
        super().__init__(**kwargs)
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")
        self.backbone = backbone
        self._preprocess = BACKBONES[backbone][1]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._preprocess(inputs)

    def get_config(self) -> dict:
        return {**super().get_config(), "backbone": self.backbone}


# --------------------------------------------------------------------------- #
# Shared blocks
# --------------------------------------------------------------------------- #


def augmentation_block(strength: float = 0.05, seed: int = SEED) -> Sequential:
    """Flip / small rotation / small zoom.

    Keras augmentation layers are inactive at inference, so this is train-only
    by construction — regularisation, *not* class balancing (the imbalance is
    handled by class weights).
    """
    return Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(strength, seed=seed),
            layers.RandomZoom(strength, seed=seed),
        ],
        name="augment",
    )


# --------------------------------------------------------------------------- #
# From-scratch CNN
# --------------------------------------------------------------------------- #


def build_baseline_cnn(
    n_classes: int,
    img_size: int = 128,
    *,
    base_filters: int = 32,
    dropout: float = 0.3,
    dense_units: int = 128,
    aug_strength: float = 0.05,
    learning_rate: float = 1e-3,
    seed: int = SEED,
) -> Model:
    """Four conv blocks with BatchNorm + MaxPool, a GAP head, softmax output.

    The defaults reproduce the Step 3.1 baseline exactly; the same function is
    the search space for Step 3.2, so the tuner can only match or beat the
    baseline — it cannot be handicapped by a space that excludes it.
    """
    filters = (base_filters, base_filters * 2, base_filters * 4, base_filters * 4)

    blocks: list[layers.Layer] = []
    for width in filters:
        blocks += [
            layers.Conv2D(width, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
        ]

    model = Sequential(
        [
            layers.Input((img_size, img_size, 3)),
            layers.Rescaling(1.0 / 255),
            augmentation_block(aug_strength, seed),
            *blocks,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout),
            layers.Dense(dense_units, activation="relu"),
            layers.Dense(n_classes, activation="softmax"),
        ],
        name="baseline_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def hyperparameter_space(hp) -> dict:
    """Map a Keras Tuner ``HyperParameters`` object onto ``build_baseline_cnn``.

    Five knobs; batch size and resolution stay fixed so the only difference
    against the baseline is what was actually tuned.
    """
    return {
        "learning_rate": hp.Float("learning_rate", 1e-4, 3e-3, sampling="log"),
        "dropout": hp.Float("dropout", 0.2, 0.5, step=0.1),
        "dense_units": hp.Choice("dense_units", [128, 256]),
        "base_filters": hp.Choice("base_filters", [24, 32, 48]),
        "aug_strength": hp.Choice("aug_strength", [0.05, 0.10, 0.15]),
    }


# --------------------------------------------------------------------------- #
# Transfer learning
# --------------------------------------------------------------------------- #


def build_transfer_model(
    backbone: str,
    n_classes: int,
    img_size: int = 128,
    *,
    dropout: float = 0.3,
    hidden_units: tuple[int, ...] = (),
    aug_strength: float = 0.05,
    learning_rate: float = 1e-3,
    seed: int = SEED,
) -> tuple[Model, Model]:
    """An ImageNet backbone with a frozen body and a small new head.

    Returns ``(model, base_model)``; the second handle is what
    :func:`unfreeze_backbone` needs for phase two.
    """
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")

    constructor, _ = BACKBONES[backbone]
    base_model = constructor(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
        pooling=None,
    )
    base_model.trainable = False  # phase 1: only the head learns

    inputs = layers.Input((img_size, img_size, 3))
    x = augmentation_block(aug_strength, seed)(inputs)
    x = BackbonePreprocessing(backbone, name="imagenet_preprocessing")(x)
    x = base_model(x, training=False)  # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    for units in hidden_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    if not hidden_units:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name=f"{backbone}_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def unfreeze_backbone(
    model: Model,
    base_model: Model,
    *,
    learning_rate: float = 1e-5,
    last_n_layers: int | None = None,
    keep_batchnorm_frozen: bool = True,
) -> Model:
    """Phase two: unfreeze and recompile at a much lower learning rate.

    BatchNorm layers stay frozen so their ImageNet running statistics are not
    destroyed by small, noisy batches — the standard fine-tuning safeguard.
    A high learning rate here would overwrite the pre-trained features that are
    the entire point of transfer learning.
    """
    base_model.trainable = True

    trainable_layers = base_model.layers if last_n_layers is None else base_model.layers[-last_n_layers:]
    frozen = set(id(layer) for layer in base_model.layers) - set(id(layer) for layer in trainable_layers)
    for layer in base_model.layers:
        if id(layer) in frozen:
            layer.trainable = False
        elif keep_batchnorm_frozen and isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
