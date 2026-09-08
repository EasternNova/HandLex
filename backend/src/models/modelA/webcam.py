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

ROOT = Path(__file__).resolve().parents[4]

DEFAULT_MODEL = (
    ROOT
    / "backend"
    / "models"
    / "hand_landmarker.task"
)

HISTORY_SIZE = 7
STABLE_COUNT = 5
CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# MediaPipe
# ============================================================

def create_detector():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(DEFAULT_MODEL)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.HandLandmarker.create_from_options(options)


# ============================================================
# Draw MediaPipe hand landmarks
# ============================================================

def draw_hand_landmarks(frame, landmarks):
    """
    Draw the 21 MediaPipe hand landmarks and their connections
    directly on the live webcam frame.
    """

    height, width = frame.shape[:2]

    # MediaPipe hand connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),# Ring
        (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
        (5, 9), (9, 13), (13, 17),            # Palm
    ]

    points = []

    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Draw connections
    for start, end in connections:

        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Draw landmark points
    for index, point in enumerate(points):

        cv2.circle(
            frame,
            point,
            5,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

        # Small landmark number
        cv2.putText(
            frame,
            str(index),
            (point[0] + 6, point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# ============================================================
# Translucent text panel
# ============================================================

def draw_translucent_text_panel(frame, text):
    """
    Draw a smaller translucent panel for the interpreted text.
    """

    overlay = frame.copy()

    # Smaller panel
    x1, y1 = 15, 315
    x2, y2 = 950, 365

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1,
    )

    # Transparency
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
        (28, 349),
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
        self.history = collections.deque(
            maxlen=HISTORY_SIZE
        )

        self.last_committed = None

    def update(self, prediction):

        self.history.append(prediction)

        counts = collections.Counter(self.history)

        stable, count = counts.most_common(1)[0]

        is_stable = count >= STABLE_COUNT

        return stable, count, is_stable

    def reset_commit(self):
        """
        Called when the hand is released.

        This allows the same sign to be committed again:

            A -> nothing -> A
        """

        self.last_committed = None


# ============================================================
# Main
# ============================================================

def main(model_path: Path):

    print("=" * 60)
    print("HandLex ModelA V5.1 — Diagnostic Webcam")
    print("Live MediaPipe Hand Landmarks Enabled")
    print("=" * 60)

    print()
    print(f"Model: {model_path}")

    package = joblib.load(model_path)

    clf = package["model"]
    classes = package["classes"]
    version = int(package["feature_version"])

    print(f"Feature version: V{version}")
    print(f"Features: {clf.n_features_in_}")
    print(f"Classes: {len(classes)}")

    print()
    print("Loading MediaPipe...")

    detector = create_detector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    smoother = TemporalSmoother()

    sentence = []

    timestamp_ms = 0

    print()
    print("Webcam started.")
    print()
    print("Controls:")
    print("  Q     = Quit")
    print("  C     = Clear sentence")
    print("  SPACE = Commit space sign")
    print("  DEL   = Delete last character")
    print()
    print(f"Temporal history: {HISTORY_SIZE} frames")
    print(f"Stable threshold: {STABLE_COUNT} frames")
    print()
    print("Diagnostic mode:")
    print("  Hand detection")
    print("  21 MediaPipe hand landmarks")
    print("  Handedness")
    print("  Raw prediction")
    print("  Confidence")
    print("  Top 3 predictions")
    print("  Stable prediction")
    print()

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                print("Could not read webcam frame.")
                break

            # Mirror webcam for natural interaction
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms += 33

            result = detector.detect_for_video(
                image,
                timestamp_ms
            )

            # ------------------------------------------------
            # HAND DETECTED
            # ------------------------------------------------

            if result.hand_landmarks:

                landmarks = result.hand_landmarks[0]

                # ------------------------------------------------
                # LIVE MEDIAPIPE LANDMARK VISUALIZATION
                # ------------------------------------------------

                draw_hand_landmarks(
                    frame,
                    landmarks,
                )

                handedness = None

                try:
                    handedness = (
                        result.handedness[0][0].category_name
                    )
                except Exception:
                    pass

                x = make_feature_vector(
                    landmarks,
                    handedness,
                    version,
                ).reshape(1, -1)

                # Raw probabilities
                probabilities = clf.predict_proba(x)[0]

                ranked = probabilities.argsort()[::-1]

                best_idx = int(ranked[0])

                raw_prediction = classes[best_idx]

                confidence = float(
                    probabilities[best_idx]
                )

                # Top 3
                top3 = []

                for idx in ranked[:3]:

                    label = classes[int(idx)]

                    probability = float(
                        probabilities[int(idx)]
                    )

                    top3.append(
                        (label, probability)
                    )

                # Temporal smoothing
                stable, stable_count, is_stable = (
                    smoother.update(raw_prediction)
                )

                # Commit only after stable prediction
                if (
                    is_stable
                    and confidence >= CONFIDENCE_THRESHOLD
                    and stable != "nothing"
                    and stable != smoother.last_committed
                ):

                    if stable == "space":

                        sentence.append(" ")

                    elif stable == "del":

                        if sentence:
                            sentence.pop()

                    else:

                        sentence.append(stable)

                    smoother.last_committed = stable

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
                    f"Handedness: {handedness}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"RAW: {raw_prediction} "
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
                        (35, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    y_position += 28

                cv2.putText(
                    frame,
                    f"STABLE: {stable} "
                    f"({stable_count}/{HISTORY_SIZE})",
                    (20, 270),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 0),
                    2,
                )

            # ------------------------------------------------
            # NO HAND
            # ------------------------------------------------

            else:

                stable, stable_count, is_stable = (
                    smoother.update("nothing")
                )

                # Hand release resets commit gate
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

            # ------------------------------------------------
            # SENTENCE
            # ------------------------------------------------

            text = "".join(sentence)

            draw_translucent_text_panel(
                frame,
                text,
            )

            cv2.imshow(
                "HandLex ModelA V5.1 Diagnostic",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("c"):
                sentence.clear()
                smoother.reset_commit()

            elif key == 32:
                sentence.append(" ")

            # Backspace / DEL
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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained ModelA .pkl file",
    )

    args = parser.parse_args()

    main(args.model)