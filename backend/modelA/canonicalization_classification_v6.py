import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path(
    "backend/artifacts/modelA/modelA_rf_v6.pkl"
)

DATA_PATH = Path(
    "backend/data/modelA/modelA_v6_full.npz"
)

# =========================================================
# LOAD
# =========================================================

package = joblib.load(MODEL_PATH)

model = package["model"]

data = np.load(
    DATA_PATH,
    allow_pickle=True
)

X = data["X"]
y = data["y"]
classes = np.asarray(data["classes"])

print()
print("MODEL A V6 CANONICALIZATION CLASSIFICATION AUDIT")
print("=" * 80)

print(f"Dataset samples : {len(X)}")
print(f"Feature count   : {X.shape[1]}")

# =========================================================
# HAND-PRESENT SAMPLES
# =========================================================

valid = X[:, 93] > 0.5

L = X[valid, :63].reshape(-1, 21, 3)
Y = y[valid]

X_original = X[valid]

print(f"Hand-present samples: {len(L)}")

# =========================================================
# CANONICALIZATION
# =========================================================

def canonicalize(hand):

    wrist = hand[0]

    index_mcp = hand[5]
    middle_mcp = hand[9]

    centered = hand - wrist

    x_axis = index_mcp - wrist

    x_norm = np.linalg.norm(x_axis)

    if x_norm < 1e-8:
        return None

    x_axis = x_axis / x_norm

    middle_direction = middle_mcp - wrist

    y_axis = (
        middle_direction
        - np.dot(
            middle_direction,
            x_axis
        ) * x_axis
    )

    y_norm = np.linalg.norm(y_axis)

    if y_norm < 1e-8:
        return None

    y_axis = y_axis / y_norm

    z_axis = np.cross(
        x_axis,
        y_axis
    )

    z_norm = np.linalg.norm(z_axis)

    if z_norm < 1e-8:
        return None

    z_axis = z_axis / z_norm

    y_axis = np.cross(
        z_axis,
        x_axis
    )

    y_axis = y_axis / np.linalg.norm(y_axis)

    canonical = np.stack(
        [
            centered @ x_axis,
            centered @ y_axis,
            centered @ z_axis
        ],
        axis=1
    )

    return canonical.astype(np.float32)


# =========================================================
# REBUILD V3 FEATURES
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


def calculate_angle(a, b, c):

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0

    cosine = np.dot(
        ba,
        bc
    ) / (
        norm_ba * norm_bc
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return float(
        np.arccos(cosine)
    )


def build_features(batch, handedness):

    output = []

    for i, hand in enumerate(batch):

        angles = np.asarray(
            [
                calculate_angle(
                    hand[a],
                    hand[b],
                    hand[c]
                )
                for a, b, c in angle_triplets
            ],
            dtype=np.float32
        )

        wrist = hand[0]

        distances = np.asarray(
            [
                np.linalg.norm(
                    hand[j] - wrist
                )
                for j in range(1, 21)
            ],
            dtype=np.float32
        )

        geometric = np.concatenate(
            [
                hand.flatten(),
                angles,
                distances
            ]
        )

        vector = np.concatenate(
            [
                geometric,
                np.array(
                    [1.0],
                    dtype=np.float32
                ),
                np.asarray(
                    [handedness[i]],
                    dtype=np.float32
                )
            ]
        )

        output.append(vector)

    return np.asarray(
        output,
        dtype=np.float32
    )


# =========================================================
# CANONICALIZE
# =========================================================

canonical = []
keep = []

for i, hand in enumerate(L):

    result = canonicalize(hand)

    if result is not None:
        canonical.append(result)
        keep.append(i)

canonical = np.asarray(
    canonical,
    dtype=np.float32
)

keep = np.asarray(
    keep
)

Y = Y[keep]
X_original = X_original[keep]

# =========================================================
# ORIGINAL PREDICTION
# =========================================================

original_predictions = model.predict(
    X_original
)

original_accuracy = np.mean(
    original_predictions == Y
)

# =========================================================
# CANONICALIZED FEATURES
# =========================================================

canonical_features = build_features(
    canonical,
    X_original[:, 94]
)

print()
print("FEATURE CHECK")
print("-" * 80)

print(
    f"Canonical feature shape: "
    f"{canonical_features.shape}"
)

# =========================================================
# V6 CLASSIFIER ON CANONICALIZED FEATURES
# =========================================================

canonical_predictions = model.predict(
    canonical_features
)

canonical_accuracy = np.mean(
    canonical_predictions == Y
)

prediction_change = np.mean(
    canonical_predictions
    != original_predictions
)

print()
print("CLASSIFICATION RESULTS")
print("-" * 80)

print(
    f"Original V6 accuracy       : "
    f"{original_accuracy * 100:.4f}%"
)

print(
    f"Canonicalized input accuracy: "
    f"{canonical_accuracy * 100:.4f}%"
)

print(
    f"Prediction change rate      : "
    f"{prediction_change * 100:.4f}%"
)

# =========================================================
# PER-CLASS ACCURACY
# =========================================================

print()
print("PER-CLASS CANONICALIZED ACCURACY")
print("-" * 80)

for class_id, class_name in enumerate(classes):

    mask = Y == class_id

    if not np.any(mask):
        continue

    accuracy = np.mean(
        canonical_predictions[mask]
        == Y[mask]
    )

    print(
        f"{str(class_name):<10} "
        f"{accuracy * 100:7.2f}%"
    )

print()
print("AUDIT COMPLETE")
print()
print("IMPORTANT:")
print("V6 classifier was NOT retrained.")
print("V6 artifact was NOT modified.")
print("This is only a compatibility experiment.")
