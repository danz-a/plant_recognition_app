# Put here: the notebook pipeline  ·  Owner: each person downloads & cleans their own

Download from Colab as `.ipynb` **with outputs** (Datei → Herunterladen → .ipynb),
clean the code/comments, then commit. Submitting `.ipynb` with visible outputs beats
`.py` conversions — a jury skims notebooks, not scripts.

## The canonical Deliverable-2 pipeline (names already correct — keep them)

- [ ] `00_setup_and_split.ipynb`
- [ ] `04_scratch_cnn.ipynb`
- [ ] `09_transfer_densenet121.ipynb`
- [ ] `11_interpretability_gradcam.ipynb`
- [ ] `12_final_test_evaluation.ipynb`
- [ ] `13_export_inference_model.ipynb`   (new — Christoph, from the export build)

## Earlier-phase notebooks (rename on download — proposal, confirm in info doc)

- [ ] `01_exploration.ipynb`        ← Step1_PlantVillage_Exploration
- [ ] `02_preprocessing.ipynb`      ← plant_recognition_preprocess_…
- [ ] `03_baseline_models.ipynb`    ← Step3_1_PlantVillage_Baseline_Models
- [ ] `05_random_forest.ipynb`      ← Step3_1_PlantVillage_RandomForest
- [ ] `06_random_forest_global6.ipynb` ← Step3_1_PlantVillage_RandomForest_Global6
- [ ] `07_cnn_tuning.ipynb`         ← Step3_2_PlantVillage_CNN_Tuning
- [ ] `08_transfer_learning.ipynb`  ← Step3_2_PlantVillage_Transfer_Learning

Numbering fills the gaps in the canonical sequence — adjust if the team prefers,
but record the final convention in Alexander's "info" doc and apply it once.

## Do NOT commit

Superseded variants (`Step3_2d_…`, `Model_DenseNet_121`, `Step3_2_…ResNet50/EfficientNet`
duplicates, `Untitled*`) — they stay in Colab/Drive as history. One training record
per model in the repo.

Delete this file afterwards.
