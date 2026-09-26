import numpy as np
from pathlib import Path

DATA_PATH = Path(
    "backend/modelA/data/modelA_v6_full.npz"
)

data = np.load(
    DATA_PATH,
    allow_pickle=True
)

X = data["X"]
y = data["y"]
classes = data["classes"]

print()
print("MODEL A V6 ORIENTATION / GEOMETRY AUDIT")
print("=" * 80)

print(f"Samples: {len(X)}")
print(f"Landmark shape: {X[:, :63].reshape(-1, 21, 3).shape}")

# ---------------------------------------------------------
# First 63 features = normalized XYZ coordinates
# ---------------------------------------------------------

landmarks = X[:, :63].reshape(-1, 21, 3)

# ---------------------------------------------------------
# Estimate hand orientation using wrist -> middle MCP
# Wrist = landmark 0
# Middle MCP = landmark 9
# ---------------------------------------------------------

wrist = landmarks[:, 0, :]
middle_mcp = landmarks[:, 9, :]

direction = middle_mcp - wrist

norm = np.linalg.norm(
    direction,
    axis=1,
    keepdims=True
)

norm[norm < 1e-8] = 1.0

direction = direction / norm

# Angles relative to camera axes.
# abs() folds positive/negative directions together.
x_angle = np.degrees(
    np.arccos(
        np.clip(
            np.abs(direction[:, 0]),
            0,
            1
        )
    )
)

y_angle = np.degrees(
    np.arccos(
        np.clip(
            np.abs(direction[:, 1]),
            0,
            1
        )
    )
)

z_angle = np.degrees(
    np.arccos(
        np.clip(
            np.abs(direction[:, 2]),
            0,
            1
        )
    )
)

print()
print("MIDDLE-FINGER AXIS ORIENTATION")
print("-" * 80)

print(
    f"X-axis angle: "
    f"mean={x_angle.mean():.2f} deg  "
    f"std={x_angle.std():.2f} deg  "
    f"min={x_angle.min():.2f} deg  "
    f"max={x_angle.max():.2f} deg"
)

print(
    f"Y-axis angle: "
    f"mean={y_angle.mean():.2f} deg  "
    f"std={y_angle.std():.2f} deg  "
    f"min={y_angle.min():.2f} deg  "
    f"max={y_angle.max():.2f} deg"
)

print(
    f"Z-axis angle: "
    f"mean={z_angle.mean():.2f} deg  "
    f"std={z_angle.std():.2f} deg  "
    f"min={z_angle.min():.2f} deg  "
    f"max={z_angle.max():.2f} deg"
)

# ---------------------------------------------------------
# Overall hand scale after normalization
# ---------------------------------------------------------

scales = np.linalg.norm(
    landmarks,
    axis=2
).max(axis=1)

print()
print("NORMALIZATION CHECK")
print("-" * 80)

print(
    f"Max wrist-distance: "
    f"mean={scales.mean():.6f}  "
    f"std={scales.std():.6f}  "
    f"min={scales.min():.6f}  "
    f"max={scales.max():.6f}"
)

# ---------------------------------------------------------
# Per-class orientation spread
# ---------------------------------------------------------

print()
print("PER-CLASS ORIENTATION SPREAD")
print("-" * 80)

for class_id, class_name in enumerate(classes):

    mask = y == class_id

    if not np.any(mask):
        continue

    print(
        f"{str(class_name):<10} "
        f"Xstd={x_angle[mask].std():6.2f} deg  "
        f"Ystd={y_angle[mask].std():6.2f} deg  "
        f"Zstd={z_angle[mask].std():6.2f} deg"
    )

print()
print("AUDIT COMPLETE")

