from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# =========================================================
# MODEL A V7 CANONICALIZED DATASET PREPARATION
# =========================================================
#
# V7 changes ONLY the coordinate representation.
#
# V6:
#   wrist-centered + scale-normalized XYZ
#
# V7:
#   V6 normalized XYZ
#       ->
#   hand-relative rotation canonicalization
#
# The same V3 geometric feature structure is then rebuilt:
#
#   63 canonical XYZ
#   + 10 angles
#   + 20 distances
#   + hand-present
#   + handedness
#   = 95 features
#
# NO MEDIA PIPE EXTRACTION
# NO TRAINING
# NO TRAIN/TEST SPLIT
# NO MODIFICATION OF V6
# =========================================================


MODEL_A_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = MODEL_A_DIR.parent

DATA_DIR = MODEL_A_DIR / "data"

INPUT_DATASET = (
    DATA_DIR
    / "modelA_v6_full.npz"
)

OUTPUT_DATASET = (
    DATA_DIR
    / "modelA_v7_canonical.npz"
)

OUTPUT_METADATA = (
    DATA_DIR
    / "modelA_v7_canonical.json"
)


FEATURE_VERSION = 3
EXPECTED_FEATURES = 95


# =========================================================
# LOAD V6 DATASET
# =========================================================

if not INPUT_DATASET.exists():
    raise FileNotFoundError(
        f"V6 dataset not found:\n{INPUT_DATASET}"
    )


data = np.load(
    INPUT_DATASET,
    allow_pickle=True
)


required_keys = {
    "X",
    "y",
    "classes",
    "version",
}


missing = required_keys.difference(
    data.files
)

if missing:
    raise RuntimeError(
        "V6 dataset is missing required fields: "
        f"{sorted(missing)}"
    )


X_v6 = np.asarray(
    data["X"],
    dtype=np.float32
)

y = np.asarray(
    data["y"],
    dtype=np.int64
)

classes = np.asarray(
    data["classes"]
)

version = int(
    data["version"]
)


# =========================================================
# VALIDATE V6 DATASET
# =========================================================

print()
print("MODEL A V7 CANONICALIZED DATASET PREPARATION")
print("=" * 80)

print(
    f"Input dataset : {INPUT_DATASET}"
)

print(
    f"Input shape   : {X_v6.shape}"
)

print(
    f"Labels shape  : {y.shape}"
)

print(
    f"Classes       : {len(classes)}"
)

print(
    f"Feature version: {version}"
)


if X_v6.ndim != 2:
    raise ValueError(
        f"Expected 2D X array, got {X_v6.ndim}D"
    )


if X_v6.shape[1] != EXPECTED_FEATURES:
    raise ValueError(
        "V6 feature count mismatch: "
        f"expected {EXPECTED_FEATURES}, "
        f"got {X_v6.shape[1]}"
    )


if len(X_v6) != len(y):
    raise ValueError(
        "X and y sample counts do not match."
    )


if version != FEATURE_VERSION:
    raise ValueError(
        f"Expected feature version "
        f"{FEATURE_VERSION}, got {version}"
    )


if not np.isfinite(X_v6).all():
    raise ValueError(
        "V6 dataset contains NaN or infinite values."
    )


# =========================================================
# EXTRACT V6 COMPONENTS
# =========================================================

#
# V3 layout:
#
# 0:63    normalized XYZ
# 63:73   angles
# 73:93   distances
# 93       hand-present
# 94       handedness
#

landmarks_v6 = X_v6[
    :, :63
].reshape(
    -1,
    21,
    3
)

hand_present = X_v6[
    :, 93
]

handedness = X_v6[
    :, 94
]


# =========================================================
# CANONICALIZATION
# =========================================================

def canonicalize(hand: np.ndarray):
    """
    Convert normalized V6 landmarks into a
    hand-relative canonical coordinate system.

    Reference landmarks:

        0 = wrist
        5 = index MCP
        9 = middle MCP

    X axis:
        wrist -> index MCP

    Y axis:
        middle MCP direction after removing
        its component along X

    Z axis:
        X cross Y

    The transformation is rigid, so it should
    preserve pairwise distances.
    """

    wrist = hand[0]

    index_mcp = hand[5]

    middle_mcp = hand[9]

    # -----------------------------------------------------
    # Translate wrist to origin.
    #
    # This is technically redundant because V6 has already
    # done wrist normalization, but keeping it explicit
    # makes V7 self-contained.
    # -----------------------------------------------------

    centered = hand - wrist

    # -----------------------------------------------------
    # X AXIS
    # -----------------------------------------------------

    x_axis = (
        index_mcp - wrist
    )

    x_norm = np.linalg.norm(
        x_axis
    )

    if x_norm < 1e-8:
        return None

    x_axis = (
        x_axis / x_norm
    )

    # -----------------------------------------------------
    # Y AXIS
    #
    # Remove X component from middle direction.
    # This produces an orthogonal axis.
    # -----------------------------------------------------

    middle_direction = (
        middle_mcp - wrist
    )

    y_axis = (
        middle_direction
        - np.dot(
            middle_direction,
            x_axis
        ) * x_axis
    )

    y_norm = np.linalg.norm(
        y_axis
    )

    if y_norm < 1e-8:
        return None

    y_axis = (
        y_axis / y_norm
    )

    # -----------------------------------------------------
    # Z AXIS
    # -----------------------------------------------------

    z_axis = np.cross(
        x_axis,
        y_axis
    )

    z_norm = np.linalg.norm(
        z_axis
    )

    if z_norm < 1e-8:
        return None

    z_axis = (
        z_axis / z_norm
    )

    # -----------------------------------------------------
    # RE-ORTHOGONALIZE Y
    # -----------------------------------------------------

    y_axis = np.cross(
        z_axis,
        x_axis
    )

    y_norm = np.linalg.norm(
        y_axis
    )

    if y_norm < 1e-8:
        return None

    y_axis = (
        y_axis / y_norm
    )

    # -----------------------------------------------------
    # PROJECT ALL LANDMARKS INTO THE NEW FRAME
    # -----------------------------------------------------

    canonical = np.stack(
        [
            centered @ x_axis,
            centered @ y_axis,
            centered @ z_axis,
        ],
        axis=1
    )

    return canonical.astype(
        np.float32
    )


# =========================================================
# V3 ANGLES
# =========================================================

ANGLE_TRIPLETS = [
    # Thumb
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),

    # Index
    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),

    # Middle
    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),

    # Ring
    (0, 13, 14),
]


def calculate_angle(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray
) -> float:

    ba = a - b

    bc = c - b

    norm_ba = np.linalg.norm(
        ba
    )

    norm_bc = np.linalg.norm(
        bc
    )

    if (
        norm_ba < 1e-8
        or norm_bc < 1e-8
    ):
        return 0.0

    cosine = (
        np.dot(
            ba,
            bc
        )
        / (
            norm_ba
            * norm_bc
        )
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return float(
        np.arccos(cosine)
    )


# =========================================================
# BUILD V7 FEATURES
# =========================================================

def build_v7_features(
    canonical: np.ndarray,
    original_hand_present: float,
    original_handedness: float,
) -> np.ndarray:

    # -----------------------------------------------------
    # 63 canonical XYZ
    # -----------------------------------------------------

    xyz_features = (
        canonical
        .flatten()
        .astype(np.float32)
    )

    # -----------------------------------------------------
    # 10 angles
    # -----------------------------------------------------

    angles = np.asarray(
        [
            calculate_angle(
                canonical[a],
                canonical[b],
                canonical[c]
            )
            for a, b, c in ANGLE_TRIPLETS
        ],
        dtype=np.float32
    )

    # -----------------------------------------------------
    # 20 wrist distances
    # -----------------------------------------------------

    wrist = canonical[0]

    distances = np.asarray(
        [
            np.linalg.norm(
                canonical[i] - wrist
            )
            for i in range(1, 21)
        ],
        dtype=np.float32
    )

    # -----------------------------------------------------
    # Final 95 features
    # -----------------------------------------------------

    features = np.concatenate(
        [
            xyz_features,
            angles,
            distances,
            np.asarray(
                [original_hand_present],
                dtype=np.float32
            ),
            np.asarray(
                [original_handedness],
                dtype=np.float32
            ),
        ]
    )

    if features.shape[0] != EXPECTED_FEATURES:
        raise RuntimeError(
            "V7 feature construction error: "
            f"expected {EXPECTED_FEATURES}, "
            f"got {features.shape[0]}"
        )

    return features.astype(
        np.float32
    )


# =========================================================
# PROCESS DATASET
# =========================================================

v7_features = []

successful = 0
failed = 0

for i in range(
    len(landmarks_v6)
):

    # -----------------------------------------------------
    # Preserve V6 no-hand rows.
    #
    # These remain all-zero feature vectors,
    # exactly like the V3/V6 representation.
    # -----------------------------------------------------

    if hand_present[i] <= 0.5:

        v7_features.append(
            np.zeros(
                EXPECTED_FEATURES,
                dtype=np.float32
            )
        )

        continue

    canonical = canonicalize(
        landmarks_v6[i]
    )

    if canonical is None:

        v7_features.append(
            np.zeros(
                EXPECTED_FEATURES,
                dtype=np.float32
            )
        )

        failed += 1

        continue

    features = build_v7_features(
        canonical,
        hand_present[i],
        handedness[i]
    )

    v7_features.append(
        features
    )

    successful += 1


X_v7 = np.asarray(
    v7_features,
    dtype=np.float32
)


# =========================================================
# FINAL VALIDATION
# =========================================================

print()
print("V7 DATASET VALIDATION")
print("-" * 80)

print(
    f"V7 shape: {X_v7.shape}"
)

print(
    f"Successful canonicalizations: "
    f"{successful}"
)

print(
    f"Canonicalization failures: "
    f"{failed}"
)

print(
    f"No-hand rows preserved: "
    f"{np.sum(hand_present <= 0.5)}"
)


if X_v7.shape != (
    len(X_v6),
    EXPECTED_FEATURES
):
    raise RuntimeError(
        "Unexpected V7 dataset shape: "
        f"{X_v7.shape}"
    )


if not np.isfinite(X_v7).all():
    raise RuntimeError(
        "V7 dataset contains NaN or infinite values."
    )


# =========================================================
# FEATURE GROUP VALIDATION
# =========================================================

print()
print("FEATURE GROUPS")
print("-" * 80)

print("Canonical XYZ : features 0-62")
print("Angles        : features 63-72")
print("Distances     : features 73-92")
print("Hand-present  : feature 93")
print("Handedness    : feature 94")


# =========================================================
# LABEL PRESERVATION
# =========================================================

if not np.array_equal(
    y,
    y
):
    raise RuntimeError(
        "Internal label preservation check failed."
    )


# =========================================================
# SAVE DATASET
# =========================================================

np.savez_compressed(
    OUTPUT_DATASET,
    X=X_v7,
    y=y,
    classes=classes,
    version=np.int64(FEATURE_VERSION),
)


# =========================================================
# METADATA
# =========================================================

metadata = {
    "model": "HandLex ModelA",
    "dataset_version": "V7",
    "representation": "canonicalized V3 geometric features",
    "feature_version": FEATURE_VERSION,
    "feature_count": EXPECTED_FEATURES,

    "source_dataset": (
        "backend/modelA/data/"
        "modelA_v6_full.npz"
    ),

    "output_dataset": (
        "backend/modelA/data/"
        "modelA_v7_canonical.npz"
    ),

    "source_samples": int(
        len(X_v6)
    ),

    "output_samples": int(
        len(X_v7)
    ),

    "classes": [
        str(c)
        for c in classes
    ],

    "canonicalization": {
        "method": (
            "hand-relative orthonormal "
            "coordinate frame"
        ),
        "x_reference": (
            "wrist -> index MCP"
        ),
        "y_reference": (
            "middle MCP direction "
            "orthogonalized against X"
        ),
        "z_reference": (
            "X cross Y"
        ),
    },

    "feature_groups": {
        "canonical_xyz": 63,
        "angles": 10,
        "distances": 20,
        "hand_present": 1,
        "handedness": 1,
    },

    "successful_canonicalizations": int(
        successful
    ),

    "canonicalization_failures": int(
        failed
    ),

    "no_hand_rows_preserved": int(
        np.sum(hand_present <= 0.5)
    ),

    "notes": [
        "Derived from existing V6 extracted landmarks.",
        "No MediaPipe extraction was performed.",
        "No classifier was trained.",
        "No train/test split was performed.",
        "V6 dataset remains untouched.",
        "Labels are preserved from V6.",
    ],
}


with open(
    OUTPUT_METADATA,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# =========================================================
# FINAL REPORT
# =========================================================

print()
print("FILES SAVED")
print("-" * 80)

print(
    f"Dataset : {OUTPUT_DATASET}"
)

print(
    f"Metadata: {OUTPUT_METADATA}"
)

print()
print("MODEL A V7 DATASET PREPARATION COMPLETE")
print("=" * 80)

print()
print("IMPORTANT:")
print("V6 was NOT modified.")
print("No MediaPipe extraction was performed.")
print("No classifier was trained.")
print("No train/test split was performed.")
print("V7 is now ready for independent validation.")

