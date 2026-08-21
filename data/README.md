# Data

The dataset is **PlantVillage** (colour version, raw and unmodified): 54,305 RGB images,
256×256 px, across **38 crop-and-condition classes** (including healthy classes).
It is **not committed** to this repository.

## Getting it

The images are downloaded from Kaggle. The exact dataset slug and download cell are in
`notebooks/00_setup_and_split.ipynb` — use that notebook, not a manual download, because
it also verifies the file inventory the split was built on.

## The split is the contract

`splits/` contains the canonical train/val/test assignment (70/15/15, stratified,
seed 42). Its fingerprint `SPLIT_ID = 9e33ec57c1ec` is derived from the MD5s of the
three CSVs and asserted at every notebook startup. If you rebuild the data folder,
the fingerprint check tells you whether your copy matches the one all results
were produced on.

## `samples/`

A handful of test-split images committed for the Streamlit demo, so the live
prediction never depends on anyone finding a photo mid-presentation.
