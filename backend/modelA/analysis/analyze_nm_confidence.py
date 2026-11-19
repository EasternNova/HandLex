
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


# Resolve paths relative to Model A
MODEL_A_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = MODEL_A_DIR / "artifacts" / "modelA_rf_v6.pkl"
DATA_PATH = MODEL_A_DIR / "data" / "modelA_v6_full.npz"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"V6 model not found: {MODEL_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"V6 dataset not found: {DATA_PATH}")

artifact = joblib.load(MODEL_PATH)
data = np.load(DATA_PATH, allow_pickle=True)

X = data["X"]
y = data["y"]
paths = data["paths"]

classes = artifact["classes"]
model = artifact["model"]

# Recreate the exact V6 80/20 stratified split
_, X_test, _, y_test, _, paths_test = train_test_split(
    X,
    y,
    paths,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

N_ID = 13
M_ID = 12

print()
print("V6 N / M CONFIDENCE ANALYSIS")
print("=" * 90)

errors = []

for i in range(len(y_test)):
    actual = int(y_test[i])
    predicted = int(predictions[i])

    if (
        (actual == N_ID and predicted == M_ID)
        or (actual == M_ID and predicted == N_ID)
    ):
        probs = probabilities[i]
        top_indices = np.argsort(probs)[::-1][:5]

        errors.append(
            (
                float(probs[predicted]),
                actual,
                predicted,
                paths_test[i],
                top_indices,
                probs,
            )
        )

# Sort by prediction confidence without comparing NumPy arrays
errors.sort(key=lambda item: item[0], reverse=True)

for confidence, actual, predicted, path, top_indices, probs in errors:
    print()
    print(f"ACTUAL     : {classes[actual]}")
    print(f"PREDICTED  : {classes[predicted]}")
    print(f"CONFIDENCE : {confidence * 100:.2f}%")
    print(f"IMAGE      : {path}")

    print("TOP 5:")

    for rank, idx in enumerate(top_indices, 1):
        print(
            f"  {rank}. "
            f"{classes[idx]:<8} "
            f"{probs[idx] * 100:6.2f}%"
        )

print()
print("=" * 90)
print(f"Total N/M errors: {len(errors)}")
print("ANALYSIS COMPLETE")
