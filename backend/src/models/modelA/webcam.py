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
# PATHS / CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[4]

DEFAULT_MODEL = (
    ROOT
    / "backend"
    / "models"
    / "hand_landmarker.task"
)

# Number of recent predictions considered for smoothing.
HISTORY_SIZE = 7

# Minimum number of identical predictions required inside
# the history window before a prediction is considered stable.
STABLE_COUNT = 5

# Minimum confidence required for a hand prediction.
CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# MEDIAPIPE HAND DETECTOR
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

    return vision.HandLandmarker.create_from_options(
        options
    )


# ============================================================
# TEMPORAL SMOOTHER
# ============================================================

class TemporalSmoother:

    def __init__(
        self,
        history_size=HISTORY_SIZE,
        stable_count=STABLE_COUNT,
    ):

        self.history = collections.deque(
            maxlen=history_size
        )

        self.stable_count = stable_count

        # Last prediction that was accepted as stable.
        self.stable_prediction = "nothing"

        # Last prediction that was committed to the sentence.
        self.last_committed = None

    def update(self, prediction):

        self.history.append(prediction)

        if len(self.history) < self.stable_count:
            return "nothing"

        counts = collections.Counter(
            self.history
        )

        candidate, count = counts.most_common(1)[0]

        if count >= self.stable_count:
            self.stable_prediction = candidate

        return self.stable_prediction

    def should_commit(self, prediction):

        if prediction == "nothing":
            return False

        if prediction == self.last_committed:
            return False

        self.last_committed = prediction

        return True

    def reset_commit_state(self):

        self.last_committed = None

    def clear(self):

        self.history.clear()
        self.stable_prediction = "nothing"
        self.last_committed = None


# ============================================================
# SENTENCE HANDLING
# ============================================================

def commit_prediction(
    prediction,
    sentence,
):

    if prediction == "nothing":
        return

    if prediction == "space":

        # Avoid multiple consecutive spaces.
        if sentence and sentence[-1] != " ":
            sentence.append(" ")

    elif prediction == "del":

        if sentence:
            sentence.pop()

    else:

        sentence.append(prediction)


# ============================================================
# MAIN WEBCAM LOOP
# ============================================================

def main(model_path: Path):

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("=" * 60)
    print("HandLex ModelA V5")
    print("=" * 60)

    print()
    print(f"Model: {model_path.resolve()}")

    package = joblib.load(model_path)

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

    if version != 3:
        print()
        print(
            "WARNING: V5 is designed "
            "for the V3 feature model."
        )

    # --------------------------------------------------------
    # CREATE DETECTOR
    # --------------------------------------------------------

    detector = create_detector()

    # --------------------------------------------------------
    # OPEN WEBCAM
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        detector.close()
        raise RuntimeError(
            "Could not open webcam"
        )

    # --------------------------------------------------------
    # V5 STATE
    # --------------------------------------------------------

    smoother = TemporalSmoother()

    sentence = []

    timestamp_ms = 0

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    print()
    print("Webcam started.")
    print()
    print("Controls:")
    print("  Q     = Quit")
    print("  C     = Clear sentence")
    print("  SPACE = Commit space sign")
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

    try:

        while True:

            # =================================================
            # READ FRAME
            # =================================================

            ok, frame = cap.read()

            if not ok:
                print(
                    "Could not read webcam frame."
                )
                break

            # =================================================
            # MIRROR FRAME
            # =================================================

            frame = cv2.flip(
                frame,
                1,
            )

            # =================================================
            # MEDIAPIPE
            # =================================================

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            timestamp_ms += 33

            result = detector.detect_for_video(
                image,
                timestamp_ms,
            )

            # =================================================
            # DEFAULT PREDICTION
            # =================================================

            pred = "nothing"
            conf = 1.0

            # =================================================
            # HAND DETECTED
            # =================================================

            if result.hand_landmarks:

                landmarks = (
                    result.hand_landmarks[0]
                )

                handedness = None

                try:

                    handedness = (
                        result
                        .handedness[0][0]
                        .category_name
                    )

                except Exception:
                    pass

                # ------------------------------------------------
                # CREATE V3 FEATURES
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
                # MODEL PREDICTION
                # ------------------------------------------------

                probs = clf.predict_proba(x)[0]

                idx = int(
                    probs.argmax()
                )

                pred = classes[idx]

                conf = float(
                    probs[idx]
                )

                # ------------------------------------------------
                # LOW CONFIDENCE → NOTHING
                # ------------------------------------------------

                if conf < CONFIDENCE_THRESHOLD:

                    pred = "nothing"

            # =================================================
            # NO HAND
            # =================================================

            else:

                pred = "nothing"
                conf = 1.0

            # =================================================
            # TEMPORAL SMOOTHING
            # =================================================

            stable = smoother.update(
                pred
            )

            # =================================================
            # COMMIT STABLE PREDICTION
            # =================================================

            if smoother.should_commit(
                stable
            ):

                commit_prediction(
                    stable,
                    sentence,
                )

            # =================================================
            # DISPLAY
            # =================================================

            sentence_text = "".join(
                sentence
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Prediction: {pred}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Confidence: {conf:.2f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # ------------------------------------------------
            # Stable prediction
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Stable: {stable}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            # ------------------------------------------------
            # Sentence
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Text: {sentence_text}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # ------------------------------------------------
            # Controls
            # ------------------------------------------------

            cv2.putText(
                frame,
                "Q: Quit   C: Clear",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            # =================================================
            # SHOW WINDOW
            # =================================================

            cv2.imshow(
                "HandLex ModelA V5",
                frame,
            )

            # =================================================
            # KEYBOARD
            # =================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ------------------------------------------------
            # QUIT
            # ------------------------------------------------

            if key == ord("q"):

                break

            # ------------------------------------------------
            # CLEAR SENTENCE
            # ------------------------------------------------

            if key == ord("c"):

                sentence.clear()

                smoother.reset_commit_state()

                print(
                    "Sentence cleared."
                )

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
            "HandLex ModelA V5 "
            "temporal smoothing webcam"
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained ModelA .pkl file",
    )

    args = parser.parse_args()

    main(
        args.model
    )