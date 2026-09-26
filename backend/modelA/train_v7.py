from pathlib import Path
import json

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# =========================================================
# MODEL A V7 TRAINING
# =========================================================

MODEL_A_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = MODEL_A_DIR.parent

DATA_PATH = (
    MODEL_A_DIR
    / "data"
    / "modelA_v7_canonical.npz"
)

ARTIFACT_DIR = (
    MODEL_A_DIR
    / "artifacts"
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "modelA_rf_v7.pkl"
)

METADATA_PATH = (
    ARTIFACT_DIR
    / "modelA_rf_v7.json"
)

EXPECTED_FEATURES = 95
EXPECTED_CLASSES = 29


# =========================================================
# LOAD DATA
# =========================================================

print()
print("MODEL A V7 TRAINING")
print("=" * 80)

print(
    f"Dataset: {DATA_PATH}"
)

data = np.load(
    DATA_PATH,
    allow_pickle=True
)

X = np.asarray(
    data["X"],
    dtype=np.float32
)

y = np.asarray(
    data["y"],
    dtype=np.int64
)

classes = np.asarray(
    data["classes"]
)

version = int(
    data["version"]
)

print(
    f"X shape: {X.shape}"
)

print(
    f"y shape: {y.shape}"
)

print(
    f"Classes: {len(classes)}"
)

print(
    f"Feature version: {version}"
)


# =========================================================
# VALIDATION
# =========================================================

if X.shape[1] != EXPECTED_FEATURES:
    raise ValueError(
        f"Expected {EXPECTED_FEATURES} features, "
        f"got {X.shape[1]}"
    )

if len(classes) != EXPECTED_CLASSES:
    raise ValueError(
        f"Expected {EXPECTED_CLASSES} classes, "
        f"got {len(classes)}"
    )

if len(X) != len(y):
    raise ValueError(
        "X and y lengths do not match."
    )

if not np.isfinite(X).all():
    raise ValueError(
        "Dataset contains NaN or infinite values."
    )

if y.min() < 0 or y.max() >= len(classes):
    raise ValueError(
        "Invalid label IDs detected."
    )


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

print()
print("CLASS DISTRIBUTION")
print("-" * 80)

for class_id, class_name in enumerate(classes):

    count = int(
        np.sum(y == class_id)
    )

    print(
        f"{str(class_name):<10} {count}"
    )


# =========================================================
# STRATIFIED SPLIT
# =========================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
)

print()
print("DATA SPLIT")
print("-" * 80)

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Test samples:     {len(X_test)}"
)


# =========================================================
# RANDOM FOREST
#
# Same configuration as V6.
# =========================================================

print()
print("MODEL")
print("-" * 80)

print("RandomForestClassifier")
print("n_estimators = 300")
print("random_state = 42")
print("n_jobs = -1")
print("class_weight = balanced")
print("max_features = sqrt")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
    max_features="sqrt",
)


# =========================================================
# TRAIN
# =========================================================

print()
print("TRAINING")
print("-" * 80)

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# =========================================================
# EVALUATION
# =========================================================

print()
print("EVALUATION")
print("-" * 80)

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"Accuracy: "
    f"{accuracy * 100:.4f}%"
)

y_test_names = np.asarray(
    [
        classes[int(label)]
        for label in y_test
    ]
)

prediction_names = np.asarray(
    [
        classes[int(label)]
        for label in predictions
    ]
)

print()
print(
    classification_report(
        y_test_names,
        prediction_names,
        labels=classes,
        zero_division=0
    )
)

matrix = confusion_matrix(
    y_test_names,
    prediction_names,
    labels=classes
)

print(
    f"Confusion matrix shape: "
    f"{matrix.shape}"
)


# =========================================================
# SAVE ARTIFACT
# =========================================================

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

package = {
    "model": model,
    "classes": classes,
    "feature_version": version,
    "feature_count": EXPECTED_FEATURES,
    "accuracy": float(accuracy),
    "model_name": "ModelA-RandomForest-V7-Canonicalized",
}

joblib.dump(
    package,
    MODEL_PATH
)


# =========================================================
# SAVE METADATA
# =========================================================

metadata = {
    "model": "HandLex ModelA",
    "model_name": "ModelA-RandomForest-V7-Canonicalized",
    "version": "V7",
    "feature_version": int(version),
    "feature_count": EXPECTED_FEATURES,
    "classes": [
        str(c)
        for c in classes
    ],
    "dataset": (
        "backend/modelA/data/"
        "modelA_v7_canonical.npz"
    ),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "accuracy": float(accuracy),
    "classifier": {
        "type": "RandomForestClassifier",
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
        "max_features": "sqrt",
    },
    "representation": {
        "xyz": 63,
        "angles": 10,
        "distances": 20,
        "hand_present": 1,
        "handedness": 1,
        "total": 95,
    },
    "notes": [
        "V7 uses hand-relative rotation canonicalization.",
        "V6 remains untouched.",
        "Same dataset samples and labels as V6.",
        "Same stratified split configuration as V6.",
        "Same Random Forest configuration as V6.",
    ],
}

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# =========================================================
# COMPLETE
# =========================================================

print()
print("=" * 80)
print("MODEL A V7 TRAINING COMPLETE")
print("=" * 80)

print()
print(
    f"Model saved: {MODEL_PATH}"
)

print(
    f"Metadata saved: {METADATA_PATH}"
)

print(
    f"V7 accuracy: {accuracy * 100:.4f}%"
)

print()
print("V6 artifact was NOT modified.")

