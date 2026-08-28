PRESENTATION_TITLE = "Plant Recognition - Leaf Disease Classification"

PROJECT_DESCRIPTION = (
    "A 38-class leaf-disease classifier on PlantVillage - "
    "DenseNet-121, macro-F1 0.9919 on the sealed test set."
)

# The model ladder shown in the demo, weakest first.
# val_f1 for the two CNNs comes from Deliverable 2, Table C.1 (validation
# macro-F1). The classical models carry their own number inside
# models/classical_spec.json, written by the export notebook (14).
MODEL_ORDER = ["logreg", "rf6", "rf160", "scratch", "densenet"]

MODEL_INFO = {
    "logreg": {
        "name": "Logistic regression",
        "sees": "6 global numbers (file size, brightness, RGB means, green fraction)",
        "val_f1": None,  # from classical_spec.json
    },
    "rf6": {
        "name": "Random Forest - 6 features",
        "sees": "the same 6 global numbers",
        "val_f1": None,  # from classical_spec.json
    },
    "rf160": {
        "name": "Random Forest - 160 features",
        "sees": "colour histograms + texture (64 px) + the 6 numbers",
        "val_f1": None,  # from classical_spec.json
    },
    "scratch": {
        "name": "Scratch CNN",
        "sees": "all pixels, 128 px, learned from zero",
        "val_f1": 0.973,
    },
    "densenet": {
        "name": "DenseNet-121 (final model)",
        "sees": "all pixels, 128 px, ImageNet pre-trained",
        "val_f1": 0.9918,
    },
}
