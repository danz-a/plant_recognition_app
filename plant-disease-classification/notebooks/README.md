# Notebooks

| Notebook | Purpose |
|---|---|
| `01_exploration.ipynb` | Data audit, class imbalance, statistical tests, the five required figures. Port the original Step 1 walkthrough here — its analysis is the argument the rest of the project rests on. |
| `02_modelling.ipynb` | Baseline CNN vs. DenseNet-121 on the identical split, with Grad-CAM. Included as a worked example of the pattern below. |
| `03_results.ipynb` | Final comparison table, per-class analysis, the validation-to-test gap. Reads the saved `artifacts/*/results.csv` rather than retraining. |

## The division of labour

**Notebooks hold the argument. The library holds the logic.**

A notebook cell should read like a sentence in the report — load the split, train
the model, read the result. If a cell defines a function, that function almost
certainly belongs in `src/plantvillage/`, because the moment a second notebook
needs it, the two copies start to drift. That is not hypothetical here: the
original notebooks disagreed on the random seed and on the augmentation strength,
which quietly made two of the models incomparable.

```python
# instead of 40 lines of pipeline setup:
split = pv_data.load_or_create_split(color_dir, class_names, paths.splits, config)
pipelines = datasets.datasets_from_split(split, name_to_index, config)

# instead of 60 lines of metrics, tables and plotting:
result = evaluation.evaluate_model(model, pipelines["val"], y_val, class_names, "Baseline CNN")
result.summary()
evaluation.plot_confusion_matrix(result)
```

## Before committing

Notebook outputs bloat diffs and can leak file paths, so strip them:

```bash
pip install nbstripout
nbstripout --install          # registers a git filter for this repository
```

Charts worth keeping belong in `docs/` as exported images, referenced from the
README — they render on GitHub without anyone opening a notebook.

## Long-running work

Training belongs in `scripts/`, not in a notebook. A script survives a dropped
Colab connection, takes arguments, and writes a `run_config.json` next to its
results. Notebooks are for reading those results and explaining them.
