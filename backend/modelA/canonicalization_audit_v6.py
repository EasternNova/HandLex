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
print("MODEL A V6 ROTATION CANONICALIZATION AUDIT")
print("=" * 80)

print(f"Samples: {len(X)}")

# ---------------------------------------------------------
# Extract normalized landmarks
# ---------------------------------------------------------

landmarks = X[:, :63].reshape(-1, 21, 3)

hand_present = X[:, 93] > 0.5

print(f"Hand-present samples: {hand_present.sum()}")

# Only analyze real detected hands.
L = landmarks[hand_present]
Y = y[hand_present]

# ---------------------------------------------------------
# Build canonical coordinate system
#
# Wrist       = landmark 0
# Index MCP   = landmark 5
# Middle MCP  = landmark 9
#
# We construct:
#
#   X axis = wrist -> index MCP
#   Y axis = middle MCP direction after removing X component
#   Z axis = X cross Y
#
# This creates a hand-relative 3D coordinate frame.
# ---------------------------------------------------------

def canonicalize(hand):

    wrist = hand[0]

    index_mcp = hand[5]
    middle_mcp = hand[9]

    # Move wrist to origin.
    centered = hand - wrist

    # -----------------------------------------------------
    # X axis
    # -----------------------------------------------------

    x_axis = index_mcp - wrist

    x_norm = np.linalg.norm(x_axis)

    if x_norm < 1e-8:
        return None

    x_axis = x_axis / x_norm

    # -----------------------------------------------------
    # Y axis
    #
    # Remove the component of middle direction that lies
    # along X. This makes X and Y perpendicular.
    # -----------------------------------------------------

    middle_direction = middle_mcp - wrist

    y_axis = (
        middle_direction
        - np.dot(middle_direction, x_axis) * x_axis
    )

    y_norm = np.linalg.norm(y_axis)

    if y_norm < 1e-8:
        return None

    y_axis = y_axis / y_norm

    # -----------------------------------------------------
    # Z axis
    # -----------------------------------------------------

    z_axis = np.cross(
        x_axis,
        y_axis
    )

    z_norm = np.linalg.norm(z_axis)

    if z_norm < 1e-8:
        return None

    z_axis = z_axis / z_norm

    # Re-orthogonalize Y to reduce numerical error.
    y_axis = np.cross(
        z_axis,
        x_axis
    )

    y_axis = y_axis / np.linalg.norm(y_axis)

    # -----------------------------------------------------
    # Transform landmarks into hand-relative coordinates.
    #
    # Each landmark is projected onto the new axes.
    # -----------------------------------------------------

    canonical = np.stack(
        [
            centered @ x_axis,
            centered @ y_axis,
            centered @ z_axis
        ],
        axis=1
    )

    return canonical.astype(np.float32)


# ---------------------------------------------------------
# Canonicalize all valid samples
# ---------------------------------------------------------

canonical_landmarks = []
valid_indices = []

for i, hand in enumerate(L):

    result = canonicalize(hand)

    if result is not None:
        canonical_landmarks.append(result)
        valid_indices.append(i)

canonical_landmarks = np.asarray(
    canonical_landmarks,
    dtype=np.float32
)

Y_canonical = Y[
    np.asarray(valid_indices)
]

print(
    f"Successfully canonicalized: "
    f"{len(canonical_landmarks)}"
)

# ---------------------------------------------------------
# Orientation measurement function
#
# We measure the wrist -> middle MCP direction.
# Before canonicalization this direction varies.
# After canonicalization it should point primarily along
# the canonical Y direction.
# ---------------------------------------------------------

def orientation_stats(batch):

    wrist = batch[:, 0, :]
    middle = batch[:, 9, :]

    direction = middle - wrist

    norm = np.linalg.norm(
        direction,
        axis=1,
        keepdims=True
    )

    norm[norm < 1e-8] = 1.0

    direction = direction / norm

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

    return (
        x_angle,
        y_angle,
        z_angle
    )


# ---------------------------------------------------------
# BEFORE
# ---------------------------------------------------------

before = orientation_stats(L)

# ---------------------------------------------------------
# AFTER
# ---------------------------------------------------------

after = orientation_stats(
    canonical_landmarks
)

print()
print("BEFORE CANONICALIZATION")
print("-" * 80)

names = ["X", "Y", "Z"]

for name, values in zip(names, before):

    print(
        f"{name}-axis angle: "
        f"mean={values.mean():.2f} deg  "
        f"std={values.std():.2f} deg  "
        f"min={values.min():.2f} deg  "
        f"max={values.max():.2f} deg"
    )

print()
print("AFTER CANONICALIZATION")
print("-" * 80)

for name, values in zip(names, after):

    print(
        f"{name}-axis angle: "
        f"mean={values.mean():.2f} deg  "
        f"std={values.std():.2f} deg  "
        f"min={values.min():.2f} deg  "
        f"max={values.max():.2f} deg"
    )

# ---------------------------------------------------------
# Per-class spread
# ---------------------------------------------------------

print()
print("PER-CLASS ORIENTATION SPREAD AFTER CANONICALIZATION")
print("-" * 80)

for class_id, class_name in enumerate(classes):

    mask = Y_canonical == class_id

    if not np.any(mask):
        continue

    print(
        f"{str(class_name):<10} "
        f"Xstd={after[0][mask].std():6.2f} deg  "
        f"Ystd={after[1][mask].std():6.2f} deg  "
        f"Zstd={after[2][mask].std():6.2f} deg"
    )

# ---------------------------------------------------------
# Geometry preservation check
# ---------------------------------------------------------

print()
print("GEOMETRY PRESERVATION CHECK")
print("-" * 80)

original_distances = np.linalg.norm(
    L[:, :, None, :] - L[:, None, :, :],
    axis=3
)

canonical_distances = np.linalg.norm(
    canonical_landmarks[:, :, None, :]
    - canonical_landmarks[:, None, :, :],
    axis=3
)

# Only compare samples successfully canonicalized.
original_distances = original_distances[
    np.asarray(valid_indices)
]

distance_difference = np.abs(
    original_distances
    - canonical_distances
)

print(
    f"Mean pairwise distance difference: "
    f"{distance_difference.mean():.8f}"
)

print(
    f"Maximum pairwise distance difference: "
    f"{distance_difference.max():.8f}"
)

print()
print("AUDIT COMPLETE")
print()
print("IMPORTANT:")
print("This experiment does NOT modify ModelA V6.")
print("No classifier was retrained.")
print("No artifact was changed.")

