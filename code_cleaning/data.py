"""Dataset acquisition and the canonical, leakage-free train/val/test split.

This module deliberately contains no TensorFlow: the split is the one object
every model in the project must agree on, so it has to be buildable and
verifiable anywhere.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from plantvillage.config import IMG_EXTENSIONS, KAGGLE_DATASET, DataConfig, Paths
from plantvillage.utils import get_logger

logger = get_logger(__name__)

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #


def download_dataset(paths: Paths, force: bool = False) -> Path:
    """Download PlantVillage from Kaggle, skipping if it is already on disk.

    Requires Kaggle credentials — either ``~/.kaggle/kaggle.json`` or the
    ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables
    (see :func:`plantvillage.utils.load_kaggle_credentials`).
    """
    target = paths.raw
    if not force and target.exists() and any(target.iterdir()):
        logger.info("dataset already present: %s", target)
        return target

    from kaggle.api.kaggle_api_extended import KaggleApi  # imported lazily

    target.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(target), unzip=True, quiet=False)
    logger.info("downloaded %s to %s", KAGGLE_DATASET, target)
    return target


def find_color_dir(root: Path, max_depth: int = 4) -> Path:
    """Breadth-first search for the ``color`` rendering inside the archive.

    The upstream zip nests ``color`` / ``grayscale`` / ``segmented`` under a
    wrapper folder whose name has changed between mirrors, so the path is
    discovered rather than hard-coded.
    """
    queue: deque[tuple[Path, int]] = deque([(Path(root), 0)])
    while queue:
        directory, depth = queue.popleft()
        if not directory.is_dir():
            continue
        if directory.name.lower() == "color":
            return directory
        if depth < max_depth:
            queue.extend((child, depth + 1) for child in sorted(directory.iterdir()) if child.is_dir())
    raise FileNotFoundError(f"no 'color' folder found under {root} (searched {max_depth} levels)")


def list_class_names(color_dir: Path) -> list[str]:
    """Class names in canonical (sorted) order — the label index everything uses."""
    names = sorted(d.name for d in Path(color_dir).iterdir() if d.is_dir())
    if not names:
        raise FileNotFoundError(f"no class folders inside {color_dir}")
    return names


def enumerate_images(color_dir: Path, class_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(paths, labels)`` for every image, in a deterministic order."""
    paths: list[str] = []
    labels: list[str] = []
    for name in class_names:
        for file in sorted((Path(color_dir) / name).iterdir()):
            if file.suffix.lower() in IMG_EXTENSIONS:
                paths.append(str(file))
                labels.append(name)
    return np.array(paths), np.array(labels)


# --------------------------------------------------------------------------- #
# The split
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Split:
    """A file-level train/val/test split, stratified and disjoint by construction."""

    train_paths: np.ndarray
    train_labels: np.ndarray
    val_paths: np.ndarray
    val_labels: np.ndarray
    test_paths: np.ndarray
    test_labels: np.ndarray

    # -- convenience ------------------------------------------------------- #

    def paths(self, subset: str) -> np.ndarray:
        return getattr(self, f"{subset}_paths")

    def labels(self, subset: str) -> np.ndarray:
        return getattr(self, f"{subset}_labels")

    def indices(self, subset: str, name_to_index: dict[str, int]) -> np.ndarray:
        """Integer labels in canonical class order."""
        return np.array([name_to_index[label] for label in self.labels(subset)], dtype="int32")

    @property
    def sizes(self) -> dict[str, int]:
        return {name: len(self.paths(name)) for name in SPLIT_NAMES}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        sizes = self.sizes
        total = sum(sizes.values())
        shares = ", ".join(f"{k}={v:,} ({v / total:.0%})" for k, v in sizes.items())
        return f"Split({shares})"

    # -- persistence ------------------------------------------------------- #

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in SPLIT_NAMES:
            frame = pd.DataFrame({"filepath": self.paths(name), "label": self.labels(name)})
            frame.to_csv(directory / f"split_{name}.csv", index=False)
        logger.info("split saved to %s", directory)
        return directory

    @classmethod
    def load(cls, directory: Path) -> "Split":
        directory = Path(directory)
        frames = {name: pd.read_csv(directory / f"split_{name}.csv") for name in SPLIT_NAMES}
        return cls(
            train_paths=frames["train"]["filepath"].to_numpy(),
            train_labels=frames["train"]["label"].to_numpy(),
            val_paths=frames["val"]["filepath"].to_numpy(),
            val_labels=frames["val"]["label"].to_numpy(),
            test_paths=frames["test"]["filepath"].to_numpy(),
            test_labels=frames["test"]["label"].to_numpy(),
        )

    @classmethod
    def exists(cls, directory: Path) -> bool:
        return all((Path(directory) / f"split_{name}.csv").exists() for name in SPLIT_NAMES)

    # -- the guarantee ----------------------------------------------------- #

    def verify(self, class_names: Sequence[str]) -> "Split":
        """Assert the two properties the whole project depends on.

        1. every class is present in every subset (stratification held);
        2. no file appears in more than one subset (no leakage).
        """
        expected = set(class_names)
        for name in SPLIT_NAMES:
            missing = expected - set(self.labels(name))
            if missing:
                raise AssertionError(f"{name} split is missing {len(missing)} classes: {sorted(missing)[:5]}")

        train, val, test = (set(self.paths(name)) for name in SPLIT_NAMES)
        if not (train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)):
            raise AssertionError("LEAKAGE: the same file appears in more than one split")

        logger.info("split verified: %s | all classes present, sets disjoint", self)
        return self


def create_split(
    paths: np.ndarray,
    labels: np.ndarray,
    config: DataConfig | None = None,
) -> Split:
    """Stratified 70/15/15 split at *file* level.

    Two calls to ``train_test_split``: first the test set is carved off the
    whole dataset, then validation off the remainder. Operating on the file
    lists (rather than on a batch stream via ``take``/``skip``) is what makes
    the result both stratified and disjoint — at ~36:1 imbalance a batch split
    can silently drop rare classes out of validation entirely.
    """
    config = config or DataConfig()
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=config.test_frac,
        stratify=labels,
        random_state=config.seed,
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=config.val_frac_of_remainder,
        stratify=train_labels,
        random_state=config.seed,
    )
    return Split(
        train_paths=train_paths,
        train_labels=train_labels,
        val_paths=val_paths,
        val_labels=val_labels,
        test_paths=test_paths,
        test_labels=test_labels,
    )


def load_or_create_split(
    color_dir: Path,
    class_names: Sequence[str],
    directory: Path,
    config: DataConfig | None = None,
) -> Split:
    """Load the canonical split if it exists, otherwise build and save it.

    Because the split is deterministic given the seed, rebuilding reproduces
    exactly the same files — the CSVs exist so a teammate on a different
    machine provably reads the same rows.
    """
    directory = Path(directory)
    if Split.exists(directory):
        logger.info("loading canonical split from %s", directory)
        return Split.load(directory).verify(class_names)

    logger.info("no split found in %s — building it", directory)
    paths, labels = enumerate_images(color_dir, class_names)
    logger.info("enumerated %d image files", len(paths))
    split = create_split(paths, labels, config).verify(class_names)
    split.save(directory)
    return split


# --------------------------------------------------------------------------- #
# Helpers used by several models
# --------------------------------------------------------------------------- #


def subsample_per_class(
    paths: Iterable[str],
    labels: Iterable[str],
    cap: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap the number of images per class — for smoke tests and search runs.

    Applied to *training* data only: validation and test are always scored in
    full, so this shortcut can never flatter a reported metric.
    """
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[str]] = {}
    for path, label in zip(paths, labels):
        by_class.setdefault(label, []).append(path)

    selected_paths: list[str] = []
    selected_labels: list[str] = []
    for label, class_paths in by_class.items():
        if len(class_paths) <= cap:
            chosen = class_paths
        else:
            picked = rng.choice(len(class_paths), size=cap, replace=False)
            chosen = [class_paths[i] for i in picked]
        selected_paths.extend(chosen)
        selected_labels.extend([label] * len(chosen))

    order = np.arange(len(selected_paths))
    rng.shuffle(order)
    return np.array(selected_paths)[order], np.array(selected_labels)[order]


def balanced_class_weights(int_labels: np.ndarray, n_classes: int) -> dict[int, float]:
    """Loss weights inversely proportional to class frequency, from TRAIN only.

    Without this the model chases the two ~5,500-image classes and ignores
    Potato-healthy (152 images).
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=int_labels,
    )
    return {index: float(weight) for index, weight in enumerate(weights)}
