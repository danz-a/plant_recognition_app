# Plant Recognition — Leaf Disease Classification

Capstone project · DataScientest / Liora · Cohort `may26bds_int`
**Team:** Adrian Bogdan RUS· Alexander DANZ · Christoph Clemens Martin NEDDENS · Scheima Sara OBEIDI — **Mentor:** Habiba El Husseiny
**Defence:** 1 September 2026

## The problem

A plant disease caught in its first week is a treatment; caught a month later it is a lost field. Expert plant pathology is scarce, slow and expensive — often least available where the crops matter most. This project builds the disease half of a phone-based diagnosis tool: a grower photographs a leaf and receives, within seconds, the crop and its likely condition.

Technically this is fine-grained image classification over **38 crop-and-condition classes** (including healthy classes — 38 classes ≠ 38 diseases) on the **PlantVillage** dataset: 54,305 colour images with a ~36:1 class imbalance.

## Headline result

| Model | Test macro-F1 | Test accuracy | Min per-class recall |
|---|---|---|---|
| **DenseNet-121** (two-phase transfer learning, 128 px) | **0.9919** | 0.9930 | 0.9481 |

Eleven models were compared — classical baselines through a tuned scratch CNN to transfer learning. DenseNet-121 is the first transfer model to genuinely beat the scratch baseline; earlier transfer-learning underperformance was traced to a silent fine-tuning failure (frozen backbone no-op), now guarded by a code-level trainable-parameter assertion. Full story in `reports/`.

## Test-set discipline

The dataset was split 70/15/15 (stratified, seed 42) **once**, before any modelling. The split is fingerprinted: `SPLIT_ID = 9e33ec57c1ec`, re-derived from the MD5s of the three CSVs in `splits/` and asserted at every notebook startup — a mismatch is a hard stop. All model selection used the validation set only; the sealed test set was evaluated **exactly once**, on the final model, under a run-lock (see the `SEAL` and `DRYRUN` artefacts in `results/`).

## Repository structure

```
├── data/        Dataset instructions + a few sample images (the dataset itself is NOT committed)
├── splits/      The canonical train/val/test split CSVs + fingerprint — the reproducibility anchor
├── models/      Final model files (inference-only) + class names + MD5 checksums
├── notebooks/   The full pipeline, exploration → split → training → interpretability → final test
├── results/     Metrics, classification reports, confusion data, run logs
├── figures/     All generated figures
├── reports/     Deliverable 1, Deliverable 2, Final Report (PDF)
└── streamlit/   The presentation & demo app
```

## Running the app locally

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit/app.py
```

The app loads the committed inference model from `models/` — no downloads, no training at runtime.

## Getting the data

The raw images are not committed. See `data/README.md` for the Kaggle source and download steps. The notebooks reproduce everything downstream of the raw archive from `splits/` + seed 42.
