import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


MODEL_PATH = Path(
    "backend/modelA/artifacts/modelA_rf_v7.pkl"
)

DATA_PATH = Path(
    "backend/modelA/data/modelA_v7_canonical.npz"
)


package = joblib.load(
    MODEL_PATH
)

model = package["model"]

data = np.load(
    DATA_PATH,
    allow_pickle=True
)

X = data["X"]
y = data["y"]
classes = np.asarray(
    data["classes"]
)


# =========================================================
# SAME SPLIT AS TRAINING
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


predictions = model.predict(
    X_test
)


accuracy = np.mean(
    predictions == y_test
)


print()
print("MODEL A V7 ERROR ANALYSIS")
print("=" * 80)

print(
    f"Test samples: {len(X_test)}"
)

print(
    f"Accuracy: {accuracy * 100:.4f}%"
)


# =========================================================
# MOST COMMON CONFUSIONS
# =========================================================

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(len(classes))
)


errors = []

for actual in range(
    len(classes)
):

    for predicted in range(
        len(classes)
    ):

        if actual == predicted:
            continue

        count = matrix[
            actual,
            predicted
        ]

        if count > 0:

            errors.append(
                (
                    int(count),
                    str(classes[actual]),
                    str(classes[predicted])
                )
            )


errors.sort(
    reverse=True
)


print()
print("MOST COMMON CONFUSIONS")
print("-" * 80)

for count, actual, predicted in errors[:40]:

    print(
        f"{actual} -> {predicted} : {count}"
    )


# =========================================================
# PER-CLASS RECALL
# =========================================================

print()
print("PER-CLASS RECALL")
print("-" * 80)

for class_id, class_name in enumerate(classes):

    support = matrix[
        class_id
    ].sum()

    correct = matrix[
        class_id,
        class_id
    ]

    if support == 0:
        recall = 0.0
    else:
        recall = (
            correct / support
        )

    print(
        f"{str(class_name):<10} "
        f"{recall * 100:7.2f}% "
        f"({support})"
    )


print()
print("AUDIT COMPLETE")

