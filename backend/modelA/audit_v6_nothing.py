from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


MODEL_A_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = MODEL_A_DIR.parent

DATA_PATH = (
    BACKEND_ROOT
    / "data"
    / "modelA_v6_full.npz"
)

MODEL_PATH = (
    BACKEND_ROOT
    / "artifacts"
    / "modelA"
    / "modelA_rf_v6.pkl"
)


print("=" * 70)
print("MODEL A V6 — NOTHING CLASS AUDIT")
print("=" * 70)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

data = np.load(DATA_PATH, allow_pickle=True)

X = data["X"]
y = data["y"]
classes = data["classes"].tolist()

print(f"Dataset samples : {len(X)}")
print(f"Feature count   : {X.shape[1]}")
print(f"Classes         : {len(classes)}")

nothing_id = classes.index("nothing")

print(f"nothing class ID: {nothing_id}")

# ------------------------------------------------------------
# Inspect nothing samples
# ------------------------------------------------------------

nothing_mask = y == nothing_id

nothing_X = X[nothing_mask]

print("\nNOTHING DATA")
print("-" * 70)
print(f"Nothing samples : {len(nothing_X)}")

zero_rows = np.all(nothing_X == 0, axis=1)

print(f"All-zero rows   : {zero_rows.sum()}")
print(
    f"Non-zero rows   : {len(nothing_X) - zero_rows.sum()}"
)

# ------------------------------------------------------------
# Same V6 train/test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

print("\nTEST SPLIT")
print("-" * 70)
print(f"Test samples    : {len(X_test)}")

# ------------------------------------------------------------
# Load V6 artifact
# ------------------------------------------------------------

package = joblib.load(MODEL_PATH)

model = package["model"]

print("\nMODEL")
print("-" * 70)
print(f"Model name      : {package.get('model_name')}")
print(f"Feature version : {package.get('feature_version')}")
print(f"Feature count   : {package.get('feature_count')}")
print(f"Classes         : {len(package.get('classes', []))}")

# ------------------------------------------------------------
# Predict
# ------------------------------------------------------------

predictions = model.predict(X_test)

# ------------------------------------------------------------
# Overall accuracy
# ------------------------------------------------------------

overall_accuracy = accuracy_score(
    y_test,
    predictions,
)

print("\nOVERALL")
print("-" * 70)
print(f"Accuracy        : {overall_accuracy * 100:.4f}%")

# ------------------------------------------------------------
# Nothing accuracy
# ------------------------------------------------------------

test_nothing = y_test == nothing_id

nothing_true = y_test[test_nothing]
nothing_pred = predictions[test_nothing]

nothing_accuracy = accuracy_score(
    nothing_true,
    nothing_pred,
)

nothing_errors = np.sum(
    nothing_pred != nothing_id
)

print("\nNOTHING CLASS")
print("-" * 70)
print(f"Test samples    : {len(nothing_true)}")
print(f"Accuracy        : {nothing_accuracy * 100:.4f}%")
print(f"Errors          : {nothing_errors}")

if nothing_errors > 0:
    print("\nNOTHING MISCLASSIFICATIONS")

    wrong_predictions = nothing_pred[
        nothing_pred != nothing_id
    ]

    for label_id in sorted(set(wrong_predictions)):
        count = np.sum(wrong_predictions == label_id)
        print(
            f"nothing -> {classes[int(label_id)]} : {count}"
        )

# ------------------------------------------------------------
# Accuracy excluding nothing
# ------------------------------------------------------------

non_nothing_test = y_test != nothing_id

y_test_without_nothing = y_test[
    non_nothing_test
]

pred_without_nothing = predictions[
    non_nothing_test
]

accuracy_without_nothing = accuracy_score(
    y_test_without_nothing,
    pred_without_nothing,
)

errors_overall = np.sum(
    predictions != y_test
)

errors_without_nothing = np.sum(
    pred_without_nothing != y_test_without_nothing
)

print("\nWITHOUT NOTHING")
print("-" * 70)
print(
    f"Test samples    : "
    f"{len(y_test_without_nothing)}"
)
print(
    f"Accuracy        : "
    f"{accuracy_without_nothing * 100:.4f}%"
)
print(f"Errors          : {errors_without_nothing}")

print("\nCOMPARISON")
print("-" * 70)

difference = (
    overall_accuracy
    - accuracy_without_nothing
) * 100

print(
    f"Overall V6      : "
    f"{overall_accuracy * 100:.4f}%"
)

print(
    f"Without nothing : "
    f"{accuracy_without_nothing * 100:.4f}%"
)

print(
    f"Difference      : "
    f"{difference:+.4f} percentage points"
)

print("\nERROR CONTRIBUTION")
print("-" * 70)
print(f"Total errors    : {errors_overall}")
print(
    f"Errors excluding "
    f"nothing         : {errors_without_nothing}"
)

print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)