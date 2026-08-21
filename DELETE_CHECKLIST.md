# Delete checklist — old files to remove from the repo

Branch `work`. Tick each box. Delete this file once done.

## 1. The wrong dataset (most important)

- [ ] `data/PlantVillage/` — **entire folder**. This is the reduced 15-class variant
      (Pepper/Potato/Tomato only); our model is trained on the full 38-class set.
      Committed images also bloat the repo. The new `.gitignore` prevents it coming back.

## 2. Superseded .py notebook conversions (repo root)

These are conversions of the *old* notebook generation. The canonical, numbered
notebooks (with outputs) replace them in `notebooks/` — keeping both would show the
jury two contradictory training records.

- [ ] `model_densenet_121.py`
- [ ] `plant_recognition_preprocess_27….py`   (full name as shown in repo)
- [ ] `plant_resnet50_kaggle_shared.py`
- [ ] `step3_1_plantvillage_baseline_m….py`
- [ ] `step3_1_plantvillage_randomfore….py`   (both random-forest variants)
- [ ] `step3_2_plantvillage_cnn_tuning….py`
- [ ] `step3_plantvillage_cnn_testset_e….py`

## 3. Duplicate requirements

- [ ] `streamlit/requirements.txt` — Streamlit Cloud reads the **root** file only.
      Two files = the classic silent-deployment-breakage trap.

## 4. Replaced root files (overwritten in Step 1 — verify, don't delete)

- [ ] old root `README.md` → replaced by the new project README
- [ ] old root `requirements.txt` → replaced by the pinned one
- [ ] old root `.gitignore` → replaced

## Explicitly KEEP

- everything inside `streamlit/` except its `requirements.txt`
  (`app.py`, `config.py`, `theme.py`, `components.py`, `sections/`, `assets/`,
  `.streamlit/`, its README)
- `.github/` if present
