
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

print()
print("V6 N / M ERROR ANALYSIS")
print("=" * 80)

for actual_id, predicted_id in [(13, 12), (12, 13)]:
    actual = classes[actual_id]
    predicted = classes[predicted_id]

    print()
    print(f"{actual} -> {predicted}")
    print("-" * 80)

    count = 0

    for i in range(len(y_test)):
        if y_test[i] == actual_id and predictions[i] == predicted_id:
            print(paths_test[i])
            count += 1

    print(f"Total: {count}")

print()
print("ANALYSIS COMPLETE")
