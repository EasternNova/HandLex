import numpy as np
from pathlib import Path

V6_PATH = Path(
    "backend/modelA/data/modelA_v6_full.npz"
)

V7_PATH = Path(
    "backend/modelA/data/modelA_v7_canonical.npz"
)

v6 = np.load(
    V6_PATH,
    allow_pickle=True
)

v7 = np.load(
    V7_PATH,
    allow_pickle=True
)

X6 = v6["X"]
y6 = v6["y"]
classes6 = np.asarray(v6["classes"])

X7 = v7["X"]
y7 = v7["y"]
classes7 = np.asarray(v7["classes"])

print()
print("MODEL A V7 DATASET INTEGRITY CHECK")
print("=" * 80)

# ---------------------------------------------------------
# Shape
# ---------------------------------------------------------

print()
print("SHAPE CHECK")
print("-" * 80)

print(f"V6 X shape: {X6.shape}")
print(f"V7 X shape: {X7.shape}")

assert X6.shape == X7.shape

print("PASS")

# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------

print()
print("LABEL CHECK")
print("-" * 80)

print(
    f"Labels identical: "
    f"{np.array_equal(y6, y7)}"
)

assert np.array_equal(
    y6,
    y7
)

print("PASS")

# ---------------------------------------------------------
# Classes
# ---------------------------------------------------------

print()
print("CLASS CHECK")
print("-" * 80)

print(
    f"Classes identical: "
    f"{np.array_equal(classes6, classes7)}"
)

assert np.array_equal(
    classes6,
    classes7
)

print("PASS")

# ---------------------------------------------------------
# No-hand rows
# ---------------------------------------------------------

print()
print("NO-HAND ROW CHECK")
print("-" * 80)

no_hand = X6[:, 93] <= 0.5

print(
    f"No-hand samples: "
    f"{no_hand.sum()}"
)

v7_no_hand = X7[no_hand]

zero_rows = np.all(
    v7_no_hand == 0,
    axis=1
)

print(
    f"V7 no-hand rows that are all zero: "
    f"{zero_rows.sum()}"
)

assert np.all(
    zero_rows
)

print("PASS")

# ---------------------------------------------------------
# Hand-present rows
# ---------------------------------------------------------

print()
print("HAND-PRESENT CHECK")
print("-" * 80)

hand_present = X6[:, 93] > 0.5

print(
    f"Hand-present samples: "
    f"{hand_present.sum()}"
)

assert np.all(
    X7[hand_present, 93] == 1.0
)

print("PASS")

# ---------------------------------------------------------
# Finite values
# ---------------------------------------------------------

print()
print("NUMERICAL VALIDITY")
print("-" * 80)

print(
    f"V7 finite: "
    f"{np.isfinite(X7).all()}"
)

assert np.isfinite(X7).all()

print("PASS")

# ---------------------------------------------------------
# Feature ranges
# ---------------------------------------------------------

print()
print("FEATURE RANGE CHECK")
print("-" * 80)

print(
    f"XYZ min: {X7[:, :63].min():.6f}"
)

print(
    f"XYZ max: {X7[:, :63].max():.6f}"
)

print(
    f"Angles min: {X7[:, 63:73].min():.6f}"
)

print(
    f"Angles max: {X7[:, 63:73].max():.6f}"
)

print(
    f"Distances min: {X7[:, 73:93].min():.6f}"
)

print(
    f"Distances max: {X7[:, 73:93].max():.6f}"
)

# ---------------------------------------------------------
# Distance preservation
#
# V6 and V7 are rigid coordinate transformations,
# so pairwise landmark distances should remain essentially
# identical.
# ---------------------------------------------------------

print()
print("DISTANCE PRESERVATION CHECK")
print("-" * 80)

sample_count = min(
    5000,
    len(X6)
)

rng = np.random.default_rng(42)

sample_indices = rng.choice(
    len(X6),
    size=sample_count,
    replace=False
)

L6 = X6[
    sample_indices,
    :63
].reshape(
    -1,
    21,
    3
)

L7 = X7[
    sample_indices,
    :63
].reshape(
    -1,
    21,
    3
)

D6 = np.linalg.norm(
    L6[:, :, None, :]
    - L6[:, None, :, :],
    axis=3
)

D7 = np.linalg.norm(
    L7[:, :, None, :]
    - L7[:, None, :, :],
    axis=3
)

difference = np.abs(
    D6 - D7
)

print(
    f"Samples checked: {sample_count}"
)

print(
    f"Mean difference: "
    f"{difference.mean():.10f}"
)

print(
    f"Maximum difference: "
    f"{difference.max():.10f}"
)

assert difference.max() < 1e-4

print("PASS")

# ---------------------------------------------------------
# Confirm representation changed
# ---------------------------------------------------------

print()
print("REPRESENTATION CHANGE CHECK")
print("-" * 80)

hand_rows = hand_present

xyz_difference = np.abs(
    X6[hand_rows, :63]
    - X7[hand_rows, :63]
)

print(
    f"Mean XYZ difference: "
    f"{xyz_difference.mean():.6f}"
)

print(
    f"Maximum XYZ difference: "
    f"{xyz_difference.max():.6f}"
)

assert xyz_difference.mean() > 1e-4

print("PASS")

# ---------------------------------------------------------
# Final
# ---------------------------------------------------------

print()
print("=" * 80)
print("V7 DATASET INTEGRITY: ALL TESTS PASSED")
print("=" * 80)

print()
print("V6 remains untouched.")
print("V7 is structurally valid.")
print("Labels are identical.")
print("No-hand samples are preserved.")
print("Geometry is preserved under transformation.")
print("The coordinate representation has changed.")
print()
print("V7 IS READY FOR TRAINING.")

