import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path("backend/artifacts/modelA/modelA_rf_v6.pkl")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]

importance = model.feature_importances_

print()
print("MODEL A V6 FEATURE IMPORTANCE AUDIT")
print("=" * 80)

print(f"Total features: {len(importance)}")
print(f"Importance sum: {importance.sum():.6f}")

# Feature groups
groups = {
    "Normalized landmarks (63)": importance[:63],
    "Joint angles (10)": importance[63:73],
    "Distances (20)": importance[73:93],
    "Hand-present flag (1)": importance[93:94],
    "Handedness (1)": importance[94:95],
}

print()
print("IMPORTANCE BY FEATURE GROUP")
print("-" * 80)

for name, values in groups.items():
    print(
        f"{name:<32} "
        f"{values.sum() * 100:7.3f}%"
    )

print()
print("TOP 20 INDIVIDUAL FEATURES")
print("-" * 80)

top_indices = np.argsort(importance)[::-1][:20]

for rank, index in enumerate(top_indices, 1):
    print(
        f"{rank:2d}. "
        f"Feature {index:2d}  "
        f"{importance[index] * 100:7.3f}%"
    )

print()
print("FEATURE GROUP RANGES")
print("-" * 80)
print("0  - 62 : normalized landmarks")
print("63 - 72 : joint angles")
print("73 - 92 : distances")
print("93      : hand-present")
print("94      : handedness")

print()
print("ANALYSIS COMPLETE")
