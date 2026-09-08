from __future__ import annotations

import argparse
import collections
from pathlib import Path

import cv2
import joblib
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from features import make_feature_vector


# ============================================================
# HandLex ModelA V5.1
# Diagnostic Webcam + Live Hand Landmarks
# ============================================================

MODEL_A_DIR = (
    Path(__file__).resolve().parent
)

BACKEND_ROOT = MODEL_A_DIR.parent

DEFAULT_LANDMARKER_MODEL = (
    BACKEND_ROOT
    / "models"
    / "pretrained"
    / "hand_landmarker.task"
)


HISTORY_SIZE = 7
STABLE_COUNT = 5
CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# MediaPipe
# ============================================================

def create_detector():

    if not DEFAULT_LANDMARKER_MODEL.exists():
        raise FileNotFoundError(
            "MediaPipe Hand Landmarker "
            "model not found:\n"
            f"{DEFAULT_LANDMARKER_MODEL}"
        )

    options = (
        vision.HandLandmarkerOptions(
            base_options=(
                python.BaseOptions(
                    model_asset_path=str(
                        DEFAULT_LANDMARKER_MODEL
                    )
                )
            ),
            running_mode=(
                vision.RunningMode.VIDEO
            ),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    return (
        vision.HandLandmarker
        .create_from_options(options)
    )


# ============================================================
# Draw MediaPipe hand landmarks
# ============================================================

def draw_hand_landmarks(
    frame,
    landmarks,
):
    """
    Draw the 21 MediaPipe hand landmarks
    and their connections.
    """

    height, width = (
        frame.shape[:2]
    )

    connections = [
        # Thumb
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        # Index
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        # Middle
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        # Ring
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        # Pinky
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        # Palm
        (5, 9),
        (9, 13),
        (13, 17),
    ]

    points = []

    for landmark in landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        points.append(
            (x, y)
        )

    # Draw connections.
    for start, end in connections:

        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Draw landmarks.
    for index, point in enumerate(
        points
    ):

        cv2.circle(
            frame,
            point,
            5,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            str(index),
            (
                point[0] + 6,
                point[1] - 6,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# ============================================================
# Translucent text panel
# ============================================================

def draw_translucent_text_panel(
    frame,
    text,
):
    """
    Draw the interpreted sentence
    on a translucent panel.
    """

    height, width = (
        frame.shape[:2]
    )

    panel_height = 50

    x1 = 15
    y2 = min(
        height - 10,
        365,
    )
    y1 = max(
        10,
        y2 - panel_height,
    )

    x2 = min(
        width - 15,
        950,
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1,
    )

    alpha = 0.55

    frame[:] = cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
    )

    cv2.putText(
        frame,
        f"TEXT: {text}",
        (28, y2 - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# Temporal smoothing
# ============================================================

class TemporalSmoother:

    def __init__(self):

        self.history = (
            collections.deque(
                maxlen=HISTORY_SIZE
            )
        )

        self.last_committed = None

    def update(
        self,
        prediction,
    ):

        self.history.append(
            prediction
        )

        if not self.history:
            return (
                prediction,
                0,
                False,
            )

        counts = (
            collections.Counter(
                self.history
            )
        )

        stable, count = (
            counts.most_common(1)[0]
        )

        is_stable = (
            count >= STABLE_COUNT
        )

        return (
            stable,
            count,
            is_stable,
        )

    def reset_commit(self):
        """
        Reset the commit gate.

        Allows the same sign to be
        committed again after the hand
        has been released.
        """

        self.last_committed = None


# ============================================================
# Main
# ============================================================

def main(
    model_path: Path,
):

    print("=" * 60)
    print(
        "HandLex ModelA V5.1 "
        "— Diagnostic Webcam"
    )
    print(
        "Live MediaPipe Hand "
        "Landmarks Enabled"
    )
    print("=" * 60)

    model_path = Path(
        model_path
    ).resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained ModelA classifier "
            f"not found:\n{model_path}"
        )

    print()
    print(
        f"Classifier model: "
        f"{model_path}"
    )

    print(
        f"Landmarker model: "
        f"{DEFAULT_LANDMARKER_MODEL}"
    )

    # --------------------------------------------------------
    # Load trained classifier
    # --------------------------------------------------------

    package = joblib.load(
        model_path
    )

    if "model" not in package:
        raise ValueError(
            "Invalid ModelA package: "
            "missing 'model'."
        )

    if "classes" not in package:
        raise ValueError(
            "Invalid ModelA package: "
            "missing 'classes'."
        )

    if "feature_version" not in package:
        raise ValueError(
            "Invalid ModelA package: "
            "missing 'feature_version'."
        )

    clf = package["model"]
    classes = package["classes"]

    version = int(
        package["feature_version"]
    )

    print(
        f"Feature version: V{version}"
    )

    print(
        f"Features: "
        f"{clf.n_features_in_}"
    )

    print(
        f"Classes: "
        f"{len(classes)}"
    )

    # --------------------------------------------------------
    # Validate feature compatibility
    # --------------------------------------------------------

    expected_features = {
        1: 63,
        2: 63,
        3: 95,
    }

    if version not in expected_features:
        raise ValueError(
            f"Unsupported feature "
            f"version: V{version}"
        )

    if (
        clf.n_features_in_
        != expected_features[version]
    ):
        raise ValueError(
            "Classifier/feature "
            "version mismatch: "
            f"V{version} expects "
            f"{expected_features[version]} "
            f"features, but classifier "
            f"expects "
            f"{clf.n_features_in_}."
        )

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    print()
    print(
        "Loading MediaPipe..."
    )

    detector = create_detector()

    # --------------------------------------------------------
    # Webcam
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        detector.close()

        raise RuntimeError(
            "Could not open webcam."
        )

    smoother = TemporalSmoother()

    sentence = []

    timestamp_ms = 0

    print()
    print(
        "Webcam started."
    )

    print()
    print("Controls:")
    print("  Q     = Quit")
    print("  C     = Clear sentence")
    print("  SPACE = Commit space")
    print("  DEL   = Delete last character")

    print()
    print(
        f"Temporal history: "
        f"{HISTORY_SIZE} frames"
    )

    print(
        f"Stable threshold: "
        f"{STABLE_COUNT} frames"
    )

    print()
    print(
        "Diagnostic mode:"
    )

    print(
        "  Hand detection"
    )

    print(
        "  21 MediaPipe hand landmarks"
    )

    print(
        "  Handedness"
    )

    print(
        "  Raw prediction"
    )

    print(
        "  Confidence"
    )

    print(
        "  Top 3 predictions"
    )

    print(
        "  Stable prediction"
    )

    print()

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                print(
                    "Could not read "
                    "webcam frame."
                )
                break

            # Mirror webcam.
            frame = cv2.flip(
                frame,
                1,
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb,
            )

            # MediaPipe VIDEO mode
            # requires increasing timestamps.
            timestamp_ms += 33

            result = (
                detector.detect_for_video(
                    image,
                    timestamp_ms,
                )
            )

            # ====================================================
            # HAND DETECTED
            # ====================================================

            if result.hand_landmarks:

                landmarks = (
                    result.hand_landmarks[0]
                )

                draw_hand_landmarks(
                    frame,
                    landmarks,
                )

                handedness = None

                try:

                    handedness = (
                        result
                        .handedness[0][0]
                        .category_name
                    )

                except (
                    IndexError,
                    AttributeError,
                ):
                    pass

                # ------------------------------------------------
                # Feature extraction
                # ------------------------------------------------

                x = make_feature_vector(
                    landmarks,
                    handedness,
                    version,
                ).reshape(
                    1,
                    -1,
                )

                # ------------------------------------------------
                # Prediction
                # ------------------------------------------------

                probabilities = (
                    clf.predict_proba(x)[0]
                )

                ranked = (
                    probabilities
                    .argsort()[::-1]
                )

                best_idx = int(
                    ranked[0]
                )

                raw_prediction = (
                    classes[best_idx]
                )

                confidence = float(
                    probabilities[
                        best_idx
                    ]
                )

                # ------------------------------------------------
                # Top 3
                # ------------------------------------------------

                top3 = []

                for idx in ranked[:3]:

                    label = classes[
                        int(idx)
                    ]

                    probability = float(
                        probabilities[
                            int(idx)
                        ]
                    )

                    top3.append(
                        (
                            label,
                            probability,
                        )
                    )

                # ------------------------------------------------
                # Temporal smoothing
                # ------------------------------------------------

                (
                    stable,
                    stable_count,
                    is_stable,
                ) = smoother.update(
                    raw_prediction
                )

                # ------------------------------------------------
                # Commit stable prediction
                # ------------------------------------------------

                if (
                    is_stable
                    and confidence
                    >= CONFIDENCE_THRESHOLD
                    and stable != "nothing"
                    and stable
                    != smoother.last_committed
                ):

                    if stable == "space":

                        sentence.append(
                            " "
                        )

                    elif stable == "del":

                        if sentence:
                            sentence.pop()

                    else:

                        sentence.append(
                            stable
                        )

                    smoother.last_committed = (
                        stable
                    )

                # ------------------------------------------------
                # DISPLAY
                # ------------------------------------------------

                cv2.putText(
                    frame,
                    "HAND: YES",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Handedness: "
                    f"{handedness}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"RAW: "
                    f"{raw_prediction} "
                    f"({confidence * 100:.1f}%)",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    "TOP 3:",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                y_position = 175

                for label, probability in top3:

                    cv2.putText(
                        frame,
                        f"{label}: "
                        f"{probability * 100:.1f}%",
                        (
                            35,
                            y_position,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    y_position += 28

                cv2.putText(
                    frame,
                    f"STABLE: "
                    f"{stable} "
                    f"({stable_count}/"
                    f"{HISTORY_SIZE})",
                    (20, 270),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 0),
                    2,
                )

            # ====================================================
            # NO HAND
            # ====================================================

            else:

                (
                    stable,
                    stable_count,
                    is_stable,
                ) = smoother.update(
                    "nothing"
                )

                # Hand release resets
                # the commit gate.
                smoother.reset_commit()

                cv2.putText(
                    frame,
                    "HAND: NO",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    "RAW: nothing",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            # ====================================================
            # SENTENCE
            # ====================================================

            text = "".join(
                sentence
            )

            draw_translucent_text_panel(
                frame,
                text,
            )

            cv2.imshow(
                "HandLex ModelA V5.1 Diagnostic",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # Q = quit
            if key == ord("q"):
                break

            # C = clear sentence
            elif key == ord("c"):

                sentence.clear()

                smoother.reset_commit()

            # SPACE
            elif key == 32:

                sentence.append(" ")

            # Backspace / Delete
            elif key in (8, 127):

                if sentence:
                    sentence.pop()

    finally:

        cap.release()

        detector.close()

        cv2.destroyAllWindows()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run HandLex ModelA "
            "diagnostic webcam inference."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help=(
            "Path to trained "
            "ModelA .pkl file"
        ),
    )

    args = parser.parse_args()

    main(
        args.model
    )