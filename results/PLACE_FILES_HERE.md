# Put here: metrics & run artefacts  ·  Owner: Christoph

Everything from Drive `plant_recognition/deliverable2/03_results/`, including:

- [ ] `final_test_evaluation_SEAL.txt` — **keep!** documents the run-lock
- [ ] all `final_test_*` files (classification report, confusion matrix/pairs,
      per-class recall, results csv)
- [ ] all `final_test_DRYRUN_*` files — **keep!** they prove the discipline:
      dry run first, sealed evaluation exactly once
- [ ] `densenet121_*` files (results, history, classification report,
      confusion matrix, per-class recall)
- [ ] `attention_*` csv files (summary, on_leaf, correct_vs_wrong, bootstrap)
- [ ] `densenet_failures.csv`, `val_predictions.csv`, `confusion_pairs.csv`
- [ ] `run_config.json`, `checkpoint_verification.csv`
      (from `plant_recognition/step3_4_interpretability/`)

The SEAL + DRYRUN files are a feature, not clutter — they are the written
evidence for the test-set-discipline answer in the Q&A.

Delete this file afterwards.
