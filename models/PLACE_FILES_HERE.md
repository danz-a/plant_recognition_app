# Put here: the model files  ·  Owner: Christoph

- [ ] `densenet121_inference.keras`  (~30 MB) — output of the **export notebook**
      (next build; strips optimizer state from the 81 MB training checkpoint and
      proves bit-identical predictions before writing)
- [ ] `scratch_cnn.keras`            (3.1 MB) — from Drive
      `plant_recognition/deliverable2/02_models/scratch_cnn.keras`
      (needed for the Grad-CAM DenseNet-vs-scratch comparison)
- [ ] `class_names.json`             — output of the export notebook
      (the 38 labels in exact training order)
- [ ] update `MD5SUMS.txt` with the values the export notebook prints

⚠ The 81 MB training checkpoint (`densenet121.keras`) stays in Drive as the
canonical training artefact. It is NOT committed. Its MD5 must be
`afe3478450c41d91a6ac1e57a2ada44d` — the export notebook asserts this.

⚠ There is a second `densenet121.keras` in Drive owned by another account —
that is an earlier run. Only the MD5 above identifies the right file.

Delete this file afterwards.
