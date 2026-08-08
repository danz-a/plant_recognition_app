"""``tf.data`` input pipelines.

One rule holds everywhere: **pixels leave this module in the 0-255 range.**
Normalisation is a layer inside the model (see :mod:`plantvillage.models`), so
training, validation and the deployed app scale images identically and a raw
JPEG can be fed straight to a saved model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import tensorflow as tf

from plantvillage.config import DataConfig
from plantvillage.data import Split

AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUFFER = 2048


def _decode(img_size: int):
    """Build the per-file decode/resize function for a given resolution."""

    def load(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [img_size, img_size])  # float32, still 0-255
        image.set_shape([img_size, img_size, 3])
        return image, label

    return load


def make_dataset(
    paths: Sequence[str],
    int_labels: np.ndarray,
    config: DataConfig | None = None,
    *,
    training: bool = False,
    cache: bool = False,
) -> tf.data.Dataset:
    """Turn a file list into a batched ``(image, label)`` dataset.

    Args:
        paths: image file paths.
        int_labels: integer labels in canonical class order.
        config: resolution and batch size.
        training: shuffle each epoch. Left ``False`` for val/test so the
            prediction order lines up with the label array.
        cache: keep decoded images in RAM. Safe for val/test, not for the
            ~38k training images on a Colab runtime.
    """
    config = config or DataConfig()
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), int_labels))
    dataset = dataset.map(_decode(config.img_size), num_parallel_calls=AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(
            min(SHUFFLE_BUFFER, len(paths)), seed=config.seed, reshuffle_each_iteration=True
        )
    return dataset.batch(config.batch_size).prefetch(AUTOTUNE)


def datasets_from_split(
    split: Split,
    name_to_index: dict[str, int],
    config: DataConfig | None = None,
    subsets: Sequence[str] = ("train", "val"),
) -> dict[str, tf.data.Dataset]:
    """Build the pipelines for the requested subsets of a :class:`Split`.

    ``test`` is not built by default — it stays sealed until the final
    evaluation, and asking for it explicitly makes that an intentional act.
    """
    config = config or DataConfig()
    return {
        subset: make_dataset(
            split.paths(subset),
            split.indices(subset, name_to_index),
            config,
            training=(subset == "train"),
            cache=(subset != "train"),
        )
        for subset in subsets
    }


def assert_raw_pixels(dataset: tf.data.Dataset) -> None:
    """Guard the contract: pixels must still be 0-255 when they leave here.

    A silent double-normalisation (dividing here *and* in the model) is the
    kind of bug that costs a training run, so it is checked rather than assumed.
    """
    images, _ = next(iter(dataset))
    peak = float(tf.reduce_max(images))
    if peak <= 1.5:
        raise AssertionError(
            f"pixels appear pre-normalised (max={peak:.3f}); the Rescaling layer lives in the model"
        )


def preview_batch(dataset: tf.data.Dataset, n: int = 9, class_names: Sequence[str] | None = None):
    """Plot a grid of images from one batch — a quick sanity check."""
    import matplotlib.pyplot as plt

    images, labels = next(iter(dataset))
    rows = int(np.ceil(n / 3))
    figure, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    for index, axis in enumerate(np.ravel(axes)):
        axis.axis("off")
        if index >= min(n, len(images)):
            continue
        axis.imshow(images[index].numpy().astype("uint8"))
        if class_names is not None:
            axis.set_title(class_names[int(labels[index])], fontsize=8)
    figure.tight_layout()
    return figure
