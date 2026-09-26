import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = Path("backend/modelA/artifacts/modelA_rf_v6.pkl")
DATA_PATH = Path("backend/modelA/data/modelA_v6_full.npz")

artifact = joblib.load(MODEL_PATH)
data = np.load(DATA_PATH, allow_pickle=True)

X = data["X"]
y = data["y"]

model = artifact["model"]

# Recreate exact V6 test split
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

# Baseline
baseline_predictions = model.predict(X_test)
baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions,
)

print()
print("MODEL A V6 PERMUTATION / GROUP ABLATION AUDIT")
print("=" * 80)

print(f"Test samples       : {len(y_test)}")
print(f"Baseline accuracy  : {baseline_accuracy * 100:.4f}%")

groups = {
    "Normalized landmarks (0-62)": list(range(0, 63)),
    "Joint angles (63-72)": list(range(63, 73)),
    "Distances (73-92)": list(range(73, 93)),
    "Hand-present (93)": [93],
    "Handedness (94)": [94],
}

rng = np.random.default_rng(42)

print()
print("GROUP PERMUTATION RESULTS")
print("-" * 80)
print("Each group is randomly shuffled across test samples.")
print("The model itself is NOT changed.")
print()

results = []

for name, indices in groups.items():

    X_perm = X_test.copy()

    # Shuffle each selected feature column independently
    for feature_index in indices:
        X_perm[:, feature_index] = rng.permutation(
            X_perm[:, feature_index]
        )

    predictions = model.predict(X_perm)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    drop = baseline_accuracy - accuracy

    results.append(
        (
            drop,
            name,
            accuracy,
        )
    )

    print(
        f"{name:<32} "
        f"accuracy={accuracy * 100:7.3f}%   "
        f"drop={drop * 100:7.3f} percentage points"
    )

print()
print("RANKED BY ACCURACY DROP")
print("-" * 80)

results.sort(reverse=True)

for rank, (drop, name, accuracy) in enumerate(results, 1):

    print(
        f"{rank}. "
        f"{name:<32} "
        f"drop={drop * 100:7.3f} pp"
    )

print()
print("ANALYSIS COMPLETE")

