"""Hand-crafted features for the classical baselines.

Two representations, deliberately kept separate:

``global6``
    The six whole-image summaries from Deliverable 1. These are the
    *documentation floor*: exploration showed they collapse onto essentially
    one lightness axis and barely separate healthy from diseased
    (:math:`\\eta^2 \\approx 1\\%`).

``rich``
    Colour histograms (RGB + HSV) plus GLCM texture, ~160 numbers — the kind of
    descriptor classical computer vision used before deep learning.

Running a linear *and* a non-linear model on ``global6`` answers the natural
question: is the ceiling the model, or the features?
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

from plantvillage.config import GLOBAL_FEATURES
from plantvillage.utils import get_logger, timer

logger = get_logger(__name__)

# Histogram / texture settings (features are computed on a cheap 64x64 copy)
FEATURE_IMAGE_SIZE = 64
RGB_BINS = 32
HSV_BINS = 16
GLCM_LEVELS = 32
GLCM_DISTANCES = (1, 2)
GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")


# --------------------------------------------------------------------------- #
# The six global features
# --------------------------------------------------------------------------- #


def global_features(path: str | Path) -> dict[str, float]:
    """File size, brightness, the three channel means and the green fraction.

    Computed on the full-resolution image, matching the Deliverable 1
    definition exactly so the numbers stay comparable across notebooks.
    """
    array = np.asarray(Image.open(path).convert("RGB")).astype("float64")
    red, green, blue = (array[:, :, index].mean() for index in range(3))
    return {
        "file_size_kb": os.path.getsize(path) / 1024,
        "brightness": float(array.mean()),
        "mean_R": float(red),
        "mean_G": float(green),
        "mean_B": float(blue),
        "green_frac": float(green / (red + green + blue)),
    }


def global_feature_table(
    paths: Sequence[str],
    labels: Sequence[str],
    cache: Path | None = None,
) -> pd.DataFrame:
    """Build (or load) a table of the six global features, one row per image."""
    if cache is not None and Path(cache).exists():
        table = pd.read_csv(cache)
        logger.info("loaded cached global features %s from %s", table.shape, cache)
        return table

    rows = []
    with timer(f"global features for {len(paths):,} images"):
        for path, label in zip(paths, labels):
            try:
                rows.append({"filepath": str(path), "label": label, **global_features(path)})
            except (OSError, ValueError) as error:  # unreadable file — skip, don't crash the run
                logger.warning("skipping %s (%s)", path, error)

    table = pd.DataFrame(rows)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache, index=False)
    return table


# --------------------------------------------------------------------------- #
# Colour histograms + GLCM texture
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureSpec:
    """Column names and their family, so importances can be summed by group."""

    names: tuple[str, ...]
    groups: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.names)

    @property
    def group_sizes(self) -> dict[str, int]:
        return {group: self.groups.count(group) for group in dict.fromkeys(self.groups)}


def rich_feature_spec() -> FeatureSpec:
    """The ~160-column descriptor: RGB hist + HSV hist + GLCM + the global six."""
    names: list[str] = []
    groups: list[str] = []

    for channel in ("R", "G", "B"):
        names += [f"rgb_{channel}_{b}" for b in range(RGB_BINS)]
        groups += ["rgb_hist"] * RGB_BINS
    for channel in ("H", "S", "V"):
        names += [f"hsv_{channel}_{b}" for b in range(HSV_BINS)]
        groups += ["hsv_hist"] * HSV_BINS
    for distance in GLCM_DISTANCES:
        names += [f"glcm_d{distance}_{prop}" for prop in GLCM_PROPERTIES]
        groups += ["glcm_texture"] * len(GLCM_PROPERTIES)
    names += list(GLOBAL_FEATURES)
    groups += ["global6"] * len(GLOBAL_FEATURES)

    return FeatureSpec(tuple(names), tuple(groups))


def _normalised_histogram(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    counts = np.histogram(values, bins=bins, range=value_range)[0].astype("float64")
    total = counts.sum()
    return counts / total if total > 0 else counts


def rich_features(path: str | Path) -> np.ndarray:
    """Colour histograms and GLCM texture for one image.

    Histograms describe *how much* of each tone is present; GLCM describes how
    rough or patterned the surface is — the closest a hand-crafted feature gets
    to "spots and lesions".
    """
    from skimage.color import rgb2gray, rgb2hsv
    from skimage.feature import graycomatrix, graycoprops

    full = np.asarray(Image.open(path).convert("RGB")).astype("float64")
    red, green, blue = (full[:, :, index].mean() for index in range(3))
    global6 = np.array(
        [
            os.path.getsize(path) / 1024,
            full.mean(),
            red,
            green,
            blue,
            green / (red + green + blue),
        ]
    )

    small = np.asarray(
        Image.fromarray(full.astype("uint8")).resize((FEATURE_IMAGE_SIZE, FEATURE_IMAGE_SIZE))
    ).astype("float64")

    rgb_hist = np.concatenate(
        [_normalised_histogram(small[:, :, c], RGB_BINS, (0, 255)) for c in range(3)]
    )
    hsv = rgb2hsv(small / 255.0)
    hsv_hist = np.concatenate(
        [_normalised_histogram(hsv[:, :, c], HSV_BINS, (0, 1)) for c in range(3)]
    )

    grey = (rgb2gray(small / 255.0) * (GLCM_LEVELS - 1)).astype("uint8")
    matrix = graycomatrix(
        grey,
        distances=list(GLCM_DISTANCES),
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    properties = {prop: graycoprops(matrix, prop).mean(axis=1) for prop in GLCM_PROPERTIES}
    glcm = np.array(
        [
            properties[prop][distance_index]
            for distance_index in range(len(GLCM_DISTANCES))
            for prop in GLCM_PROPERTIES
        ]
    )

    return np.concatenate([rgb_hist, hsv_hist, glcm, global6]).astype("float32")


def rich_feature_matrix(
    paths: Sequence[str],
    labels: Sequence[str],
    cache: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (or load) the ``rich`` feature matrix. Extraction is the slow part."""
    if cache is not None and Path(cache).exists():
        stored = np.load(cache, allow_pickle=True)
        logger.info("loaded cached features %s from %s", stored["X"].shape, cache)
        return stored["X"], stored["y"]

    vectors: list[np.ndarray] = []
    kept: list[str] = []
    with timer(f"rich features for {len(paths):,} images"):
        for path, label in zip(paths, labels):
            try:
                vectors.append(rich_features(path))
                kept.append(label)
            except (OSError, ValueError) as error:
                logger.warning("skipping %s (%s)", path, error)

    X = np.vstack(vectors)
    y = np.array(kept)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y)
    return X, y
