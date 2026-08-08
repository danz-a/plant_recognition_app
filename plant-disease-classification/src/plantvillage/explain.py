"""Explainability: Grad-CAM and SHAP.

These answer a question the metrics cannot: *where* is the model looking? For
PlantVillage that matters more than usual — every photo sits on a plain studio
background, so a model can score well by learning the background rather than
the lesion. A heatmap centred on the leaf is evidence against that shortcut;
one drifting into the background is the lab-to-field risk made visible.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


def find_last_conv_layer(model: Model) -> str:
    """Name of the deepest Conv2D layer — the default Grad-CAM target.

    The last convolution is where spatial layout and semantic meaning are both
    still present; anything after it has been pooled away.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError(f"no Conv2D layer found in {model.name!r}")


def _split_at_backbone(model: Model, backbone_name: str) -> tuple[Model, list[layers.Layer]]:
    """Return the backbone as a sub-model plus the head layers stacked on it."""
    backbone = model.get_layer(backbone_name)
    head: list[layers.Layer] = []
    seen = False
    for layer in model.layers:
        if seen:
            head.append(layer)
        if layer.name == backbone_name:
            seen = True
    if not seen:
        raise ValueError(f"{backbone_name!r} is not a layer of {model.name!r}")
    return backbone, head


def gradcam_heatmap(
    image_batch: np.ndarray,
    model: Model,
    last_conv_layer: str | None = None,
    backbone_name: str | None = None,
    class_index: int | None = None,
) -> np.ndarray:
    """Grad-CAM heatmap for one image, normalised to ``[0, 1]``.

    Args:
        image_batch: a single image with a leading batch axis, ``(1, H, W, 3)``.
        model: the trained classifier.
        last_conv_layer: target layer name; discovered automatically if omitted.
        backbone_name: for transfer models, the name of the nested backbone
            layer (e.g. ``"resnet50"``), since the target conv lives inside it.
        class_index: which class to explain. Defaults to the predicted class.
    """
    if backbone_name is not None:
        backbone, head = _split_at_backbone(model, backbone_name)
        target = last_conv_layer or find_last_conv_layer(backbone)
        feature_extractor = Model(backbone.inputs, backbone.get_layer(target).output)

        with tf.GradientTape() as tape:
            feature_maps = feature_extractor(image_batch)
            tape.watch(feature_maps)
            activations = feature_maps
            for layer in head:
                activations = layer(activations)
            index = tf.argmax(activations[0]) if class_index is None else class_index
            score = activations[:, index]
    else:
        target = last_conv_layer or find_last_conv_layer(model)
        grad_model = Model(model.inputs, [model.get_layer(target).output, model.output])
        with tf.GradientTape() as tape:
            feature_maps, predictions = grad_model(image_batch)
            index = tf.argmax(predictions[0]) if class_index is None else class_index
            score = predictions[:, index]

    gradients = tape.gradient(score, feature_maps)
    pooled = tf.reduce_mean(gradients, axis=(0, 1, 2))

    heatmap = tf.squeeze(feature_maps[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)
    peak = tf.math.reduce_max(heatmap)
    heatmap = heatmap / peak if peak > 0 else heatmap
    return heatmap.numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    title: str = "Grad-CAM",
):
    """Plot the original image next to the heatmap blended over it."""
    import matplotlib.pyplot as plt

    picture = np.asarray(image, dtype="float32")
    if picture.max() > 1.0:
        picture = picture / 255.0

    resized = tf.image.resize(heatmap[..., np.newaxis][np.newaxis, ...], picture.shape[:2])
    resized = np.squeeze(resized.numpy())
    coloured = plt.get_cmap("jet")(resized)[:, :, :3]
    blended = np.clip(coloured * alpha + picture * (1 - alpha), 0.0, 1.0)

    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, content, label in zip(axes, [picture, blended], ["original", title]):
        axis.imshow(content)
        axis.set_title(label)
        axis.axis("off")
    figure.tight_layout()
    return figure


def shap_image_plot(
    model: Model,
    images: np.ndarray,
    class_names: Sequence[str],
    max_evaluations: int = 500,
    top_classes: int = 4,
):
    """SHAP attributions for a handful of images.

    Red marks pixels pushing *towards* a class, blue pixels pushing away.
    Expensive — a few images at a time is the realistic budget.
    """
    import shap

    masker = shap.maskers.Image("inpaint_telea", images[0].shape)
    explainer = shap.Explainer(model, masker, output_names=list(class_names))
    values = explainer(
        images,
        max_evals=max_evaluations,
        outputs=shap.Explanation.argsort.flip[:top_classes],
    )
    shap.image_plot(values)
    return values
