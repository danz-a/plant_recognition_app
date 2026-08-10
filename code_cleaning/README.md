# Plant Disease Classification — PlantVillage

Classifying 38 crop-disease classes from 54,305 leaf photographs: exploration,
classical baselines, a from-scratch CNN, hyper-parameter tuning, transfer
learning and explainability.

The project is built around one claim, and the code is organised to make it
falsifiable: **the disease signal is local, so global summary features cannot
work and a convolutional model is required.** Every model is scored on the same
split by the same code, so the comparison table can be read at face value.

---

## Why macro-F1, not accuracy

The dataset is imbalanced ~36:1 — Orange/Citrus-greening has 5,507 images,
healthy Potato has 152. A classifier that ignores every rare class still posts a
respectable accuracy, which is why the headline metric here is **macro-F1**,
supported by **per-class recall** and a row-normalised **confusion matrix**.

The pipeline is built to keep that measurement honest:

| Decision | Why |
|---|---|
| **File-level stratified split**, saved to CSV | A batch `take`/`skip` split is neither stratified nor provably disjoint. At 36:1 rare classes can vanish from validation entirely. |
| **Split verified, not assumed** | `Split.verify()` raises if a class is missing from a subset or a file appears twice. |
| **Class weights from the training labels only** | Corrects the imbalance without leaking validation statistics into training. |
| **Normalisation as a model layer** | Training, validation and deployment scale identically, and a saved model accepts a raw JPEG. |
| **Augmentation as Keras layers** | Inactive at inference by construction — no accidental augmentation of validation data. |
| **Test set sealed until the end** | Opened once, by `scripts/evaluate.py`, after model selection is finished. |

## Results

Validation-set macro-F1, all models on the identical split:

| Model | Features | macro-F1 | What it establishes |
|---|---|---|---|
| Dummy (`most_frequent`, `stratified`) | — | ~0.00 | The no-skill floor, and the gap between accuracy and macro-F1. |
| Logistic regression | 6 global features | near floor | The signal is not *linearly* separable from global colour. |
| Random forest | 6 global features | near floor | Nor *non-linearly* — so the ceiling is the features, not the model. |
| Random forest | RGB/HSV histograms + GLCM | ↑ | Hand-crafted texture recovers part of the signal. |
| **CNN (from scratch)** | raw pixels | **high** | Reading local structure is what the problem needed. |
| CNN, tuned (Bayesian search) | raw pixels | ≈ baseline | The baseline architecture was already well-chosen. |
| DenseNet-121 / ResNet-50 (fine-tuned) | raw pixels | **highest** | ImageNet features transfer well to leaf textures. |

> Fill in your own numbers after a full run — `artifacts/*/results.csv` holds them.
> They are not committed, because a number without the run that produced it is a claim, not evidence.

## Quickstart

```bash
git clone https://github.com/<your-user>/plant-disease-classification.git
cd plant-disease-classification
pip install -e ".[tuning,explain,stats]"

# Kaggle credentials: place kaggle.json in ~/.kaggle/, or export
# KAGGLE_USERNAME and KAGGLE_KEY (on Colab, use Secrets — see below).

python scripts/prepare_data.py        # download + build the canonical split
python scripts/train_baselines.py     # dummy, logistic regression, random forest
python scripts/train_cnn.py           # the from-scratch baseline CNN
python scripts/tune_cnn.py            # Bayesian hyper-parameter search
python scripts/train_transfer.py --backbone densenet121
python scripts/evaluate.py --run cnn_baseline   # opens the sealed test set
```

Every script takes `--smoke-test` or a reduced budget for a two-minute dry run
before committing to a full training job. `make help` lists the shortcuts.

### On Colab or Kaggle

```python
!pip install -q git+https://github.com/<your-user>/plant-disease-classification.git
from plantvillage.utils import load_kaggle_credentials, describe_hardware
load_kaggle_credentials()   # reads KAGGLE_USERNAME / KAGGLE_KEY from Colab Secrets
print(describe_hardware())  # confirm the GPU is attached
```

Set `PLANTVILLAGE_HOME` to relocate data and artifacts (e.g. to a mounted Drive
folder) — no code changes needed.

## Repository layout

```
src/plantvillage/         The library — every notebook and script imports from here
├── config.py             Paths, seed, split ratios, run configuration
├── data.py               Download, class discovery, the canonical split (no TensorFlow)
├── datasets.py           tf.data pipelines; pixels leave here as 0-255
├── features.py           Classical features: the global six, histograms, GLCM texture
├── models.py             Baseline CNN, tunable variant, transfer backbones
├── training.py           Callbacks, fit wrapper, the two-phase transfer recipe
├── evaluation.py         Macro-F1, per-class recall, confusion matrix, comparisons
├── explain.py            Grad-CAM and SHAP
└── utils.py              Seeding, logging, Colab credentials, artifacts

scripts/                  Reproducible entry points, one per experiment
notebooks/                Narrative: exploration, modelling, results
tests/                    Split correctness and the evaluation contract
docs/                     Method notes and the refactoring map
```

## Design notes

**The library holds the logic; notebooks hold the story.** The nine original
notebooks each re-implemented the folder search, the split, the input pipeline,
the class weights and the whole evaluation block. That is roughly 200 duplicated
lines per notebook, and — more seriously — five opportunities for two models to
be silently compared on different data. Those pieces now live in one place, and
a notebook is the argument built on top of them.

**`data.py` imports no TensorFlow.** The split is the one object every model
must agree on, so it stays buildable, testable and verifiable in any
environment, including CI without a GPU.

**The split is persisted, not just seeded.** It is deterministic given the seed,
so it *would* reproduce — but the CSVs mean a teammate on another machine can
prove they read the same rows rather than trusting that they did.

**Evaluation is a function, not a copied cell.** `evaluate_predictions` returns
an `EvaluationResult` carrying the metrics, the per-class table and the
confusion matrix. Every model, classical or deep, goes through it.

## Testing

```bash
pytest -q      # split stratification, leakage detection, determinism, metrics
ruff check src scripts tests
```

The tests build a synthetic imbalanced dataset in a temp folder, so they run in
seconds without downloading anything.

## Known limitations

* **Lab conditions.** Every PlantVillage photo has a plain background and even
  lighting. A high validation score confirms stability within that
  distribution, not that the model survives a phone photo taken in a field. The
  Grad-CAM overlays in `explain.py` exist partly to check whether the model is
  leaning on the background.
* **128×128 by default.** Downsampling from the native 256×256 costs fine
  lesion detail — a conscious speed trade-off. `--img-size 256` is the
  comparison worth running.
* **Class entanglement.** Five crops appear with only one condition (Raspberry,
  Blueberry and Soybean are all healthy; Squash and Orange all diseased), so
  per-crop healthy-vs-diseased is ill-posed. The task is framed as flat 38-way
  classification for that reason.

## Data

[PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
— 54,305 images, 38 classes, three renderings (`color`, `grayscale`,
`segmented`). This project uses `color`; the exploration notebook confirms the
three renderings contain the same photographs.

## License

MIT — see `LICENSE`.
