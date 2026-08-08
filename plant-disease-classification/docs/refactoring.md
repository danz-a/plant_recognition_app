# From nine notebooks to one library

This document maps the original notebooks onto the refactored code — useful when
reviewing the change, and as the record of what was consolidated and why.

## Where each notebook went

| Original notebook | Now lives in |
|---|---|
| `step1_PlantVillage_Exploration_Walkthrough.ipynb` | `notebooks/01_exploration.ipynb` — kept as narrative; helpers moved to `data.py` / `features.py` |
| `step2_plant_recognition_preprocess_270626_revised.ipynb` | `data.py` (split) + `datasets.py` (pipelines) |
| `Step3_1_PlantVillage_Baseline_Models.ipynb` | `scripts/train_baselines.py` + `scripts/train_cnn.py` |
| `Step3_1_PlantVillage_RandomForest.ipynb` | `scripts/train_baselines.py --models rf --features rich` |
| `Step3_1_PlantVillage_RandomForest_Global6.ipynb` | `scripts/train_baselines.py --models rf --features global6` |
| `Step3_2_PlantVillage_CNN_Tuning.ipynb` | `scripts/tune_cnn.py` |
| `Step3_PlantVillage_CNN_TestSet_Evaluation.ipynb` | `scripts/evaluate.py` |
| `step3_Model_DenseNet_121.ipynb` | `scripts/train_transfer.py --backbone densenet121` |
| `step3_plant_resnet50_kaggle_shared.ipynb` | `scripts/train_transfer.py --backbone resnet50` + `explain.py` |

## What was duplicated

Each of these appeared in **six to nine** notebooks, character for character:

| Duplicated block | Now |
|---|---|
| `locate()` — breadth-first search for `color/` | `data.find_color_dir` |
| `enumerate_files()` + `train_test_split` ×2 + verification | `data.load_or_create_split`, `Split.verify` |
| `cap_per_class()` | `data.subsample_per_class` |
| `make_ds()` / `make_dataset()` | `datasets.make_dataset` |
| `compute_class_weight` block | `data.balanced_class_weights` |
| macro-F1 + per-class recall + `classification_report` + confusion heatmap | `evaluation.evaluate_model` |
| Learning-curve plotting | `evaluation.plot_learning_curves` |
| Colab Secrets / Kaggle credential cell | `utils.load_kaggle_credentials` |
| Kaggle download with skip-guard | `data.download_dataset` |

Roughly 1,400 duplicated lines collapsed into about 350 shared ones.

## Inconsistencies the consolidation surfaced

Deduplication is not only tidier — it caught real divergences between notebooks
that would have made the comparison table misleading.

1. **Two different seeds.** The Step 3 notebooks used `SEED = 42`; the Step 2
   preprocessing and the ResNet-50 notebook used `seed = 123`. Same split logic,
   *different splits* — so the ResNet-50 numbers were never comparable to the
   CNN baseline. There is now one seed, in `config.SEED`.

2. **Two different augmentation strengths.** Rotation and zoom were `0.05` in the
   baseline CNN and `0.1` in the preprocessing and ResNet-50 notebooks.
   `models.augmentation_block(strength=...)` makes this an explicit argument.

3. **Normalisation applied inconsistently.** The baseline used
   `Rescaling(1/255)` inside the model; DenseNet added an ImageNet
   `Normalization` layer; ResNet-50 mapped `preprocess_input` over the dataset
   and commented out the rescaling with the note "caused problems". The last
   variant means the saved model is *not* self-contained — feeding it a raw
   image silently produces wrong predictions. `models.BackbonePreprocessing`
   puts the backbone's own pre-processing inside the model, serialisably.

4. **ResNet-50's test evaluation ran twice on differently-processed data** —
   once on `test_ds` and again on `test_ds.map(preprocess_input)` — because it
   was unclear which was correct. With normalisation inside the model, the
   question disappears.

5. **Two model-persistence conventions.** Filenames such as `resnet50_4.keras`,
   `resnet50_5.keras`, `best_resnet50_2.keras` mixed with `joblib.dump` of a
   Keras model. Artifacts now follow one layout: `artifacts/<run>/model.keras`
   plus `run_config.json`, `results.csv`, per-class recalls and the confusion
   matrix.

6. **Evaluation reached into globals.** `evaluate_model()` in the DenseNet
   notebook read `val_ds`, `yi_val`, `class_names` and `n_classes` from the
   surrounding namespace, and its driver cell used `try: model ... except
   NameError` to detect whether training had run. That is a cell-execution-order
   dependency: run the cells out of sequence and it silently evaluates the wrong
   thing. Everything is a parameter now.

## Conventions

* **Configuration is data, not code.** `DataConfig` and `TrainConfig` are frozen
  dataclasses written next to every run as `run_config.json`.
* **Artifacts are never committed.** `.gitignore` excludes `data/` and
  `artifacts/`; the repository holds the code that reproduces them.
* **Every public function has a docstring saying *why*.** What the code does is
  visible; the reasoning behind a choice is not.
* **Failures are loud.** Leakage, missing classes and pre-normalised pixels
  raise rather than warn — a silent data bug is worth more than a crash.
