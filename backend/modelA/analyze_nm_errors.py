import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("backend/artifacts/modelA/modelA_rf_v6.pkl")
DATA_PATH = Path("backend/data/modelA/modelA_v6_full.npz")

artifact = joblib.load(MODEL_PATH)
data = np.load(DATA_PATH, allow_pickle=True)

X = data["X"]
y = data["y"]
paths = data["paths"]
classes = artifact["classes"]
model = artifact["model"]

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
