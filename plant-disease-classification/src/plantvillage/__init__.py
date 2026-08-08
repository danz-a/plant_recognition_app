"""PlantVillage disease classification — shared library for the whole project.

Every notebook and script imports from here, so the dataset split, the
pre-processing and the evaluation contract are defined exactly once.

Typical use::

    from plantvillage import Paths, DataConfig, data, datasets, evaluation

    paths = Paths.default()
    color_dir = data.find_color_dir(paths.raw)
    classes = data.list_class_names(color_dir)
    split = data.load_or_create_split(color_dir, classes, paths.splits)
"""

from plantvillage.config import (
    GLOBAL_FEATURES,
    IMG_EXTENSIONS,
    KAGGLE_DATASET,
    N_CLASSES,
    SEED,
    DataConfig,
    Paths,
    TrainConfig,
)
from plantvillage.data import Split
from plantvillage.evaluation import EvaluationResult

__all__ = [
    "GLOBAL_FEATURES",
    "IMG_EXTENSIONS",
    "KAGGLE_DATASET",
    "N_CLASSES",
    "SEED",
    "DataConfig",
    "EvaluationResult",
    "Paths",
    "Split",
    "TrainConfig",
]

__version__ = "1.0.0"
