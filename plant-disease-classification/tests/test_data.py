"""Tests for the split — the one object every model must agree on.

These run without TensorFlow and without the dataset: a synthetic folder tree
is enough to prove stratification, disjointness and reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from plantvillage.config import DataConfig
from plantvillage.data import (
    Split,
    balanced_class_weights,
    create_split,
    enumerate_images,
    find_color_dir,
    list_class_names,
    subsample_per_class,
)

CLASSES = [f"Crop{i}___condition" for i in range(4)]
COUNTS = [200, 120, 60, 20]  # deliberately imbalanced, like the real 36:1


@pytest.fixture
def dataset_root(tmp_path):
    """A fake archive: wrapper / color / <class> / <n images>."""
    color = tmp_path / "plantvillage dataset" / "color"
    for name, count in zip(CLASSES, COUNTS):
        folder = color / name
        folder.mkdir(parents=True)
        for index in range(count):
            (folder / f"img_{index:04d}.jpg").write_bytes(b"fake")
    (tmp_path / "plantvillage dataset" / "grayscale").mkdir()
    return tmp_path


def test_find_color_dir_searches_nested_folders(dataset_root):
    assert find_color_dir(dataset_root).name == "color"


def test_find_color_dir_raises_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_color_dir(tmp_path)


def test_class_names_are_sorted(dataset_root):
    names = list_class_names(find_color_dir(dataset_root))
    assert names == sorted(names) == CLASSES


def test_enumerate_images_finds_every_file(dataset_root):
    color = find_color_dir(dataset_root)
    paths, labels = enumerate_images(color, CLASSES)
    assert len(paths) == sum(COUNTS)
    assert set(labels) == set(CLASSES)


def test_split_is_stratified_and_leak_free(dataset_root):
    color = find_color_dir(dataset_root)
    paths, labels = enumerate_images(color, CLASSES)
    split = create_split(paths, labels, DataConfig()).verify(CLASSES)

    total = sum(split.sizes.values())
    assert total == sum(COUNTS)
    assert split.sizes["train"] / total == pytest.approx(0.70, abs=0.01)
    assert split.sizes["val"] / total == pytest.approx(0.15, abs=0.01)

    # Stratification: each class keeps roughly its share in every subset.
    for subset in ("train", "val", "test"):
        subset_labels = split.labels(subset)
        for name, count in zip(CLASSES, COUNTS):
            share = np.mean(subset_labels == name)
            assert share == pytest.approx(count / sum(COUNTS), abs=0.05)


def test_split_is_deterministic(dataset_root):
    color = find_color_dir(dataset_root)
    paths, labels = enumerate_images(color, CLASSES)
    first = create_split(paths, labels, DataConfig())
    second = create_split(paths, labels, DataConfig())
    assert np.array_equal(first.train_paths, second.train_paths)
    assert np.array_equal(first.test_paths, second.test_paths)


def test_verify_detects_leakage(dataset_root):
    color = find_color_dir(dataset_root)
    paths, labels = enumerate_images(color, CLASSES)
    split = create_split(paths, labels, DataConfig())
    leaking = Split(
        train_paths=split.train_paths,
        train_labels=split.train_labels,
        # a complete val set, plus ten rows copied straight out of train
        val_paths=np.concatenate([split.val_paths, split.train_paths[:10]]),
        val_labels=np.concatenate([split.val_labels, split.train_labels[:10]]),
        test_paths=split.test_paths,
        test_labels=split.test_labels,
    )
    with pytest.raises(AssertionError, match="LEAKAGE"):
        leaking.verify(CLASSES)


def test_split_roundtrips_through_csv(dataset_root, tmp_path):
    color = find_color_dir(dataset_root)
    paths, labels = enumerate_images(color, CLASSES)
    split = create_split(paths, labels, DataConfig())
    split.save(tmp_path / "splits")

    assert Split.exists(tmp_path / "splits")
    restored = Split.load(tmp_path / "splits")
    assert np.array_equal(split.val_paths, restored.val_paths)


def test_subsample_caps_each_class():
    paths = [f"{i}.jpg" for i in range(300)]
    labels = ["a"] * 200 + ["b"] * 100
    capped_paths, capped_labels = subsample_per_class(paths, labels, cap=50)
    assert len(capped_paths) == 100
    assert sum(capped_labels == "a") == 50
    assert sum(capped_labels == "b") == 50


def test_class_weights_favour_rare_classes():
    labels = np.array([0] * 100 + [1] * 10)
    weights = balanced_class_weights(labels, n_classes=2)
    assert weights[1] > weights[0]
    assert weights[1] / weights[0] == pytest.approx(10.0, rel=0.01)
