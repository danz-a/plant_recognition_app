"""Download PlantVillage and build the canonical split.

Run this once. Every other script and notebook reads the CSVs it writes, which
is what guarantees all models are trained and scored on byte-identical data.

    python scripts/prepare_data.py
"""

from __future__ import annotations

import argparse

from plantvillage import DataConfig, Paths
from plantvillage import data as pv_data
from plantvillage.utils import get_logger, load_kaggle_credentials, set_seed

logger = get_logger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=None, help="project root (default: ~/plant_recognition)")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-split", action="store_true", help="rebuild the split even if CSVs exist")
    args = parser.parse_args()

    set_seed()
    load_kaggle_credentials()

    paths = Paths.default(args.base).create()
    pv_data.download_dataset(paths, force=args.force_download)

    color_dir = pv_data.find_color_dir(paths.raw)
    class_names = pv_data.list_class_names(color_dir)
    logger.info("found %d classes in %s", len(class_names), color_dir)

    if args.force_split:
        images, labels = pv_data.enumerate_images(color_dir, class_names)
        split = pv_data.create_split(images, labels, DataConfig()).verify(class_names)
        split.save(paths.splits)
    else:
        split = pv_data.load_or_create_split(color_dir, class_names, paths.splits)

    logger.info("ready: %s", split)
    logger.info("split CSVs: %s", paths.splits)


if __name__ == "__main__":
    main()
