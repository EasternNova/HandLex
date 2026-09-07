from __future__ import annotations

import math
import numpy as np


def _xyz(landmarks):
    """
    Convert MediaPipe's 21 landmarks into a (21, 3) float32 array.
    """
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32,
    )


def normalize_landmarks(landmarks):
    """
    V2 normalization:
    - move wrist (landmark 0) to the origin
    - scale using the maximum wrist-to-landmark distance
    """
    xyz = _xyz(landmarks)

    wrist = xyz[0].copy()
    xyz = xyz - wrist

    distances = np.linalg.norm(xyz, axis=1)
    scale = float(np.max(distances))

    if scale > 1e-8:
        xyz = xyz / scale

    return xyz


def _angle(a, b, c):
    """
    Angle ABC in radians.
    """
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(math.acos(cosine))


def engineered_features(landmarks):
    """
    V3 features:
    - 63 normalized landmark coordinates
    - 10 finger joint angles
    - 20 wrist-to-landmark distances

    Total = 93 features.
    """
    xyz = normalize_landmarks(landmarks)

    # Finger joint angle definitions.
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

    angles = [
        _angle(xyz[a], xyz[b], xyz[c])
        for a, b, c in angle_triplets
    ]

    # Distances from wrist to landmarks 1..20.
    wrist = xyz[0]
    distances = [
        float(np.linalg.norm(xyz[i] - wrist))
        for i in range(1, 21)
    ]

    return np.concatenate(
        [
            xyz.flatten(),
            np.asarray(angles, dtype=np.float32),
            np.asarray(distances, dtype=np.float32),
        ]
    ).astype(np.float32)


def make_feature_vector(
    landmarks=None,
    handedness=None,
    version=1,
):
    """
    Create the feature vector used by ModelA.

    Parameters
    ----------
    landmarks:
        MediaPipe's 21 hand landmarks, or None when no hand is detected.

    handedness:
        MediaPipe handedness information. Used by V3.

    version:
        1 = raw normalized MediaPipe x/y/z coordinates (63)
        2 = wrist-relative + scale-normalized coordinates (63)
        3 = V2 + angles + distances + hand-present + handedness (95)
    """

    if landmarks is None:
        if version == 3:
            # 93 geometric features
            # + 1 hand-present flag
            # + 1 handedness flag
            return np.zeros(95, dtype=np.float32)

        return np.zeros(63, dtype=np.float32)

    if version == 1:
        # MediaPipe coordinates are already normalized to the image.
        return _xyz(landmarks).flatten().astype(np.float32)

    if version == 2:
        return normalize_landmarks(landmarks).flatten().astype(np.float32)

    if version == 3:
        geometric = engineered_features(landmarks)

        # Hand is present.
        hand_present = np.array([1.0], dtype=np.float32)

        # Right = 1, Left = 0.
        handedness_value = 0.0

        if handedness is not None:
            try:
                name = str(handedness).lower()

                if "right" in name:
                    handedness_value = 1.0
                elif "left" in name:
                    handedness_value = 0.0
            except Exception:
                pass

        handedness_feature = np.array(
            [handedness_value],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                geometric,
                hand_present,
                handedness_feature,
            ]
        ).astype(np.float32)

    raise ValueError(
        f"Unsupported feature version: {version}. "
        "Expected 1, 2, or 3."
    )