import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

MODEL_PATH = Path("backend/artifacts/modelA/modelA_rf_v6.pkl")
DATA_PATH = Path("backend/data/modelA/modelA_v6_full.npz")

artifact = joblib.load(MODEL_PATH)
data = np.load(DATA_PATH, allow_pickle=True)

X = data["X"]
y = data["y"]

classes = artifact["classes"]
model = artifact["model"]

# Recreate the exact V6 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

predictions = model.predict(X_test)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(len(classes)),
)

print()
print("MODEL A V6 REAL TEST CONFUSION ANALYSIS")
print("=" * 80)
print(f"Total dataset samples : {len(y)}")
print(f"Test samples          : {len(y_test)}")
print(f"Accuracy              : {accuracy_score(y_test, predictions):.6f}")
print(f"Accuracy (%)          : {accuracy_score(y_test, predictions) * 100:.2f}%")

print()
print("MOST COMMON CONFUSIONS")
print("-" * 80)

pairs = []

for i in range(len(classes)):
    for j in range(len(classes)):
        if i != j and cm[i, j] > 0:
            pairs.append(
                (
                    int(cm[i, j]),
                    str(classes[i]),
                    str(classes[j]),
                )
            )

pairs.sort(reverse=True)

if pairs:
    for count, actual, predicted in pairs[:30]:
        print(f"{actual:>10} -> {predicted:<10}: {count}")
else:
    print("No misclassifications found.")

print()
print("PER-CLASS RECALL")
print("-" * 80)

report = classification_report(
    y_test,
    predictions,
    labels=np.arange(len(classes)),
    target_names=classes,
    zero_division=0,
    output_dict=True,
)

for class_name in classes:
    recall = report[str(class_name)]["recall"] * 100
    support = report[str(class_name)]["support"]

    print(
        f"{str(class_name):<10}: "
        f"{recall:6.2f}% recall "
        f"(support={int(support)})"
    )

print()
print("ANALYSIS COMPLETE")
