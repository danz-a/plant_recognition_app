"""Small cross-cutting helpers: seeding, logging, credentials, JSON artifacts."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from plantvillage.config import SEED

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """A logger that prints once, cleanly, in notebooks and scripts alike."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


logger = get_logger(__name__)


def set_seed(seed: int = SEED) -> int:
    """Seed python, numpy and (if installed) TensorFlow in one call."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
    return seed


def load_kaggle_credentials() -> bool:
    """Populate the Kaggle env vars from Colab Secrets when running on Colab.

    Keeping the key in Secrets means it never lives in a committed notebook.
    Off Colab this is a no-op and the local ``~/.kaggle/kaggle.json`` is used.
    """
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except ImportError:
        return False

    os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    logger.info("Kaggle credentials loaded from Colab Secrets")
    return True


def describe_hardware() -> str:
    """One-line GPU report — the first thing to check in a fresh runtime."""
    try:
        import tensorflow as tf
    except ImportError:
        return "TensorFlow not installed"

    gpus = tf.config.list_physical_devices("GPU")
    device = gpus[0].name if gpus else "none (enable Runtime > GPU)"
    return f"TensorFlow {tf.__version__} | GPU: {device}"


def save_json(payload: dict[str, Any], path: Path) -> Path:
    """Write a JSON artifact, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


@contextmanager
def timer(label: str) -> Iterator[None]:
    """Log how long a block took — used around every training call."""
    start = time.time()
    yield
    logger.info("%s finished in %.0fs", label, time.time() - start)
