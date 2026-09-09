import numpy as np
import joblib
from pathlib import Path

# =========================================================
# PATHS
# =========================================================

MODEL_PATH = Path(
    "backend/artifacts/modelA/modelA_rf_v6.pkl"
)

DATA_PATH = Path(
    "backend/data/modelA/modelA_v6_full.npz"
)

# =========================================================
# LOAD MODEL
# =========================================================

package = joblib.load(MODEL_PATH)

model = package["model"]
classes = np.asarray(package["classes"])

# =========================================================
# LOAD DATA
# =========================================================

data = np.load(
    DATA_PATH,
    allow_pickle=True
)

X = data["X"]
y = data["y"]

print()
print("MODEL A V6 ROTATION-SENSITIVITY AUDIT")
print("=" * 80)

print(f"Dataset samples : {len(X)}")
print(f"Feature count   : {X.shape[1]}")
print(f"Classes         : {len(classes)}")

# =========================================================
# USE ONLY SAMPLES WITH A DETECTED HAND
#
# V3 has:
#   0-62  = normalized XYZ landmarks
#   63-72 = angles
#   73-92 = distances
#   93    = hand-present
#   94    = handedness
#
# We rotate the 3D landmarks and rebuild the geometric
# features while keeping the original model unchanged.
# =========================================================

landmarks = X[:, :63].reshape(-1, 21, 3)

hand_present = X[:, 93] > 0.5

valid = hand_present

landmarks = landmarks[valid]
y_valid = y[valid]

print(f"Hand-present samples: {len(landmarks)}")

# =========================================================
# ANGLE DEFINITIONS
# Same 10 angle definitions used by ModelA V3.
# =========================================================

angle_triplets = [
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),

    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),

    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),

    (0, 13, 14),
]

# =========================================================
# REBUILD V3 GEOMETRIC FEATURES
# =========================================================

def angle(a, b, c):

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0

    cosine = np.dot(ba, bc) / (
        norm_ba * norm_bc
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return np.arccos(cosine)


def build_features(batch):

    output = []

    for xyz in batch:

        angles = np.asarray(
            [
                angle(
                    xyz[a],
                    xyz[b],
                    xyz[c]
                )
                for a, b, c in angle_triplets
            ],
            dtype=np.float32
        )

        wrist = xyz[0]

        distances = np.asarray(
            [
                np.linalg.norm(
                    xyz[i] - wrist
                )
                for i in range(1, 21)
            ],
            dtype=np.float32
        )

        geometric = np.concatenate(
            [
                xyz.flatten(),
                angles,
                distances
            ]
        )

        # Preserve original V6 feature structure.
        # Hand-present = 1.
        # Handedness = original feature 94.
        output.append(
            np.concatenate(
                [
                    geometric,
                    np.array(
                        [1.0],
                        dtype=np.float32
                    ),
                ]
            )
        )

    # IMPORTANT:
    # Add original handedness feature afterward.
    geometric_features = np.asarray(
        output,
        dtype=np.float32
    )

    return geometric_features


# =========================================================
# ROTATION
# =========================================================

def rotation_matrix(
    axis,
    degrees
):

    radians = np.radians(degrees)

    c = np.cos(radians)
    s = np.sin(radians)

    if axis == "x":

        return np.array(
            [
                [1, 0, 0],
                [0, c, -s],
                [0, s, c],
            ],
            dtype=np.float32
        )

    if axis == "y":

        return np.array(
            [
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c],
            ],
            dtype=np.float32
        )

    if axis == "z":

        return np.array(
            [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ],
            dtype=np.float32
        )

    raise ValueError(
        f"Unknown rotation axis: {axis}"
    )


def rotate_landmarks(
    batch,
    axis,
    degrees
):

    R = rotation_matrix(
        axis,
        degrees
    )

    rotated = np.matmul(
        batch,
        R.T
    )

    return rotated


# =========================================================
# BASELINE
# =========================================================

# Reconstruct valid original V3 features.
#
# We use the original X rows directly for baseline so
# there is absolutely no reconstruction difference.

X_valid_original = X[valid]

baseline_predictions = model.predict(
    X_valid_original
)

baseline_accuracy = np.mean(
    baseline_predictions == y_valid
)

print()
print("BASELINE")
print("-" * 80)

print(
    f"Accuracy on hand-present samples: "
    f"{baseline_accuracy * 100:.4f}%"
)

# =========================================================
# TEST ROTATIONS
# =========================================================

angles_to_test = [
    15,
    30,
    45,
    60,
]

axes = [
    "x",
    "y",
    "z",
]

print()
print("ROTATION RESULTS")
print("-" * 80)

for axis in axes:

    print()
    print(f"ROTATION AXIS: {axis.upper()}")
    print("-" * 80)

    for degrees in angles_to_test:

        rotated = rotate_landmarks(
            landmarks,
            axis,
            degrees
        )

        rotated_features = build_features(
            rotated
        )

        # Add original handedness feature.
        rotated_features = np.concatenate(
            [
                rotated_features[:, :94],
                X_valid_original[:, 94:95]
            ],
            axis=1
        )

        predictions = model.predict(
            rotated_features
        )

        accuracy = np.mean(
            predictions == y_valid
        )

        prediction_stability = np.mean(
            predictions == baseline_predictions
        )

        print(
            f"{degrees:>2} deg  | "
            f"accuracy={accuracy * 100:7.3f}%  | "
            f"same-as-original={prediction_stability * 100:7.3f}%"
        )

print()
print("AUDIT COMPLETE")
print()
print("IMPORTANT:")
print("This experiment changes only the input landmarks.")
print("The trained V6 classifier is NOT modified.")
print("No retraining occurred.")
