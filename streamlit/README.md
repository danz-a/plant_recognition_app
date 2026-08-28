# Streamlit demo

One page: a leaf goes in, five models predict - weakest first.

```text
streamlit/
├── app.py            entry point (no navigation - the demo is the app)
├── config.py         title + the model ladder (names, what each model sees, val macro-F1 of the CNNs)
├── theme.py          colours / CSS
├── components.py     section header
├── model_utils.py    cached, MD5-verified loaders for all five models + predict_all()
├── features.py       the 6-number and 160-number classical feature extractors (Step 3.1 code, unchanged)
├── gradcam_utils.py  live Grad-CAM for DenseNet-121
└── sections/demo.py  the page
```

Models (all in `../models/`, listed in `MD5SUMS.txt`):

| key | file | produced by |
|---|---|---|
| logreg | `logreg_global6.joblib` | notebook 14 (exact Step 3.1 recipe) |
| rf6 | `rf_global6_demo.joblib` | notebook 14 (demo copy, size-capped) |
| rf160 | `rf160_demo.joblib` | notebook 14 (demo copy, size-capped) |
| scratch | `scratch_cnn.keras` | notebook 04 |
| densenet | `densenet121_inference.keras` | notebook 13 |

Nothing is trained or downloaded at runtime. Run locally with
`pip install -r ../requirements.txt` then `streamlit run app.py`.
