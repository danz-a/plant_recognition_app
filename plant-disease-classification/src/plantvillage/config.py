"""Single source of truth for paths, constants and run configuration.

Nothing in this module imports TensorFlow, so it stays importable in any
environment (CI, a plain laptop, a notebook without a GPU runtime).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants that must never diverge between notebooks
# --------------------------------------------------------------------------- #

SEED: int = 42
"""One seed for python / numpy / tensorflow. Same seed -> same split -> comparable numbers."""

KAGGLE_DATASET: str = "abdallahalidev/plantvillage-dataset"
"""The untouched upstream mirror. Deliverable 1 documents the provenance."""

N_CLASSES: int = 38
"""38 crop___condition classes in the `color` rendering."""

IMG_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
)

GLOBAL_FEATURES: tuple[str, ...] = (
    "file_size_kb",
    "brightness",
    "mean_R",
    "mean_G",
    "mean_B",
    "green_frac",
)
"""The six whole-image summaries from Deliverable 1 (Figures 3-5)."""

ENV_HOME: str = "PLANTVILLAGE_HOME"
"""Optional environment variable to relocate the data/artifact root."""


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Paths:
    """Filesystem layout of the project.

    The default root is ``~/plant_recognition`` and can be overridden with the
    ``PLANTVILLAGE_HOME`` environment variable — which is what makes the same
    code run unchanged on Colab, Kaggle and a local machine.
    """

    base: Path
    raw: Path
    splits: Path
    artifacts: Path

    @classmethod
    def default(cls, base: str | Path | None = None) -> "Paths":
        root = Path(base or os.environ.get(ENV_HOME) or Path.home() / "plant_recognition")
        return cls(
            base=root,
            raw=root / "data" / "raw",
            splits=root / "artifacts" / "splits",
            artifacts=root / "artifacts",
        )

    def run(self, name: str) -> Path:
        """Return (and create) a per-experiment artifact folder."""
        directory = self.artifacts / name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def create(self) -> "Paths":
        """Create every directory in the layout."""
        for directory in (self.raw, self.splits, self.artifacts):
            directory.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------- #
# Run configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DataConfig:
    """How images become batched tensors. Shared by every model."""

    img_size: int = 128
    batch_size: int = 32
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = SEED

    def __post_init__(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")

    @property
    def val_frac_of_remainder(self) -> float:
        """Validation fraction *after* the test set has been carved off."""
        return self.val_frac / (self.train_frac + self.val_frac)


@dataclass(frozen=True)
class TrainConfig:
    """Training budget and callback behaviour."""

    epochs: int = 20
    learning_rate: float = 1e-3
    early_stopping_patience: int = 4
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-5
    class_weights: bool = True
    seed: int = SEED


@dataclass
class RunSummary:
    """Everything needed to reproduce a run, written next to its artifacts."""

    name: str
    data: DataConfig
    train: TrainConfig
    extra: dict = field(default_factory=dict)

    def save(self, directory: Path, filename: str = "run_config.json") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / filename
        payload = {
            "name": self.name,
            "data": asdict(self.data),
            "train": asdict(self.train),
            **self.extra,
        }
        target.write_text(json.dumps(payload, indent=2, default=str))
        return target
