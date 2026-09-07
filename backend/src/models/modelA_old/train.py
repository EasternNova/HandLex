from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# HandLex ModelA - Landmark Classifier
# ============================================================

MODEL_A_DIR = Path(__file__).resolve().parent

DEFAULT_OUTPUT = MODEL_A_DIR / "modelA_rf_v1.pkl"


EXPECTED_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "del",
    "nothing",
    "space",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path: Path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    print()
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    print(f"Path: {path.resolve()}")

    data = np.load(
        path,
        allow_pickle=True
    )

    required = [
        "X",
        "y",
        "classes",
        "version",
    ]

    for key in required:
        if key not in data:
            raise ValueError(
                f"Dataset missing required field: {key}"
            )

    X = data["X"]
    y = data["y"]

    # classes is stored as a NumPy object/scalar array.
    classes = data["classes"].tolist()

    # prepare_data.py stores version as a scalar.
    # Therefore .item() is required.
    version = int(data["version"].item())

    print(f"X shape      : {X.shape}")
    print(f"y shape      : {y.shape}")
    print(f"Classes      : {len(classes)}")
    print(f"Version      : {version}")

    return X, y, classes, version


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(
    X,
    y,
    classes,
    version,
):

    print()
    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    print(f"Feature matrix : {X.shape}")
    print(f"Labels         : {y.shape}")
    print(f"Classes        : {len(classes)}")
    print(f"Feature version: {version}")

    # --------------------------------------------------------
    # Basic shape validation
    # --------------------------------------------------------

    if X.ndim != 2:
        raise ValueError(
            f"X must be 2D. Got {X.ndim}D."
        )

    if y.ndim != 1:
        raise ValueError(
            f"y must be 1D. Got {y.ndim}D."
        )

    if len(X) != len(y):
        raise ValueError(
            f"X/y size mismatch: "
            f"{len(X)} vs {len(y)}"
        )

    # --------------------------------------------------------
    # Feature validation
    # --------------------------------------------------------

    if version == 1:

        expected_features = 21 * 3

        if X.shape[1] != expected_features:
            raise ValueError(
                f"V1 expects {expected_features} features, "
                f"but dataset has {X.shape[1]}."
            )

    elif version == 2:

        expected_features = 21 * 3

        if X.shape[1] != expected_features:
            raise ValueError(
                f"V2 expects {expected_features} features, "
                f"but dataset has {X.shape[1]}."
            )

    elif version == 3:

        expected_features = 95

        if X.shape[1] != expected_features:
            raise ValueError(
                f"V3 expects {expected_features} features, "
                f"but dataset has {X.shape[1]}."
            )

    else:

        raise ValueError(
            f"Unsupported feature version: {version}"
        )

    # --------------------------------------------------------
    # Numerical validation
    # --------------------------------------------------------

    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(
            f"X must contain numerical values. "
            f"Got dtype: {X.dtype}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "X contains NaN or infinite values."
        )

    # --------------------------------------------------------
    # Class validation
    # --------------------------------------------------------

    if len(classes) != 29:
        raise ValueError(
            f"Expected 29 classes, "
            f"got {len(classes)}."
        )

    if classes != EXPECTED_CLASSES:
        raise ValueError(
            "Class order mismatch.\n\n"
            f"Found:\n{classes}\n\n"
            f"Expected:\n{EXPECTED_CLASSES}"
        )

    # --------------------------------------------------------
    # Label validation
    # --------------------------------------------------------

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Dataset must contain at least two classes."
        )

    unknown_labels = [
        label for label in np.unique(y)
        if label not in classes
    ]

    if unknown_labels:
        raise ValueError(
            f"Unknown labels found: {unknown_labels}"
        )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    print()
    print("Samples per class:")

    for label, count in zip(
        unique,
        counts
    ):
        print(
            f"  {str(label):8s}: {count}"
        )

    # --------------------------------------------------------
    # Minimum samples check
    # --------------------------------------------------------

    minimum_count = int(counts.min())

    if minimum_count < 2:
        raise ValueError(
            "At least one class has fewer than 2 samples. "
            "Stratified train/test splitting is not possible."
        )

    print()
    print(f"Minimum samples in one class: {minimum_count}")

    print()
    print("Dataset validation: PASSED")


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_random_forest(
    X_train,
    y_train,
):

    print()
    print("=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    print()
    print("Configuration:")
    print("  n_estimators : 300")
    print("  random_state : 42")
    print("  n_jobs       : -1")
    print("  class_weight : balanced")
    print("  max_features : sqrt")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
    )

    model.fit(
        X_train,
        y_train
    )

    print()
    print("Random Forest training: COMPLETE")

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    classes,
):

    print()
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print(
        f"Test accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print("Classification report:")
    print()

    report = classification_report(
        y_test,
        predictions,
        labels=classes,
        zero_division=0,
    )

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=classes,
    )

    print("Confusion matrix shape:")
    print(matrix.shape)

    return (
        accuracy,
        report,
        matrix,
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    classes,
    version,
    accuracy,
    output_path,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    package = {
        "model": model,
        "classes": classes,
        "feature_version": version,
        "feature_count": int(
            model.n_features_in_
        ),
        "accuracy": float(
            accuracy
        ),
        "model_name": "ModelA-RandomForest",
    }

    with open(
        output_path,
        "wb"
    ) as f:

        pickle.dump(
            package,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print()
    print("=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(
        f"Path    : "
        f"{output_path.resolve()}"
    )

    print(
        f"Features: "
        f"{model.n_features_in_}"
    )

    print(
        f"Classes : "
        f"{len(classes)}"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    output_path,
    data_path,
    version,
    accuracy,
    classes,
    train_size,
    test_size,
):

    metadata_path = Path(
        output_path
    ).with_suffix(".json")

    if version == 1:
        feature_count = 63
    elif version == 2:
        feature_count = 63
    elif version == 3:
        feature_count = 95
    else:
        feature_count = None

    metadata = {
        "model": "ModelA",
        "classifier": "RandomForest",
        "feature_version": version,
        "feature_count": feature_count,
        "classes": classes,
        "num_classes": len(classes),
        "total_samples": (
            train_size + test_size
        ),
        "training_samples": train_size,
        "test_samples": test_size,
        "test_accuracy": float(
            accuracy
        ),
        "dataset": str(
            Path(data_path).resolve()
        ),
        "random_state": 42,
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
    )

    print(
        f"Metadata: "
        f"{metadata_path.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train HandLex ModelA "
            "landmark classifier"
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to prepared .npz dataset",
    )

    parser.add_argument(
        "--classifier",
        choices=["rf"],
        default="rf",
        help="Classifier",
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
        help="Output model path",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HandLex ModelA - Training")
    print("=" * 60)

    print()
    print(
        "Dataset:",
        Path(args.data).resolve()
    )

    print(
        "Classifier:",
        args.classifier
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    X, y, classes, version = load_data(
        Path(args.data)
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validate_data(
        X,
        y,
        classes,
        version,
    )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Test samples:     "
        f"{len(X_test)}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    if args.classifier == "rf":

        model = train_random_forest(
            X_train,
            y_train,
        )

    else:

        raise ValueError(
            f"Unsupported classifier: "
            f"{args.classifier}"
        )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    accuracy, report, matrix = (
        evaluate_model(
            model,
            X_test,
            y_test,
            classes,
        )
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    save_model(
        model,
        classes,
        version,
        accuracy,
        args.output,
    )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    save_metadata(
        args.output,
        args.data,
        version,
        accuracy,
        classes,
        len(X_train),
        len(X_test),
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL A V1 TRAINING COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Final test accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()
    print(
        f"Model file: "
        f"{Path(args.output).resolve()}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()