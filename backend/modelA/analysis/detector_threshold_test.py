
"""
HandLex - MediaPipe Detector Threshold Sensitivity Test

Tests failed ASL images using different detection/presence
thresholds without changing the original landmarker.py.

No classifier inference.
No model retraining.
No modification of original images.
"""

from pathlib import Path
import csv

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "pretrained"
    / "hand_landmarker.task"
)

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "ASL"
    / "asl_alphabet_test"
    / "asl_alphabet_test"
)

FAILED_CSV = (
    PROJECT_ROOT
    / "backend"
    / "modelA"
    / "analysis"
    / "detection_audit"
    / "failed_detection_images.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "backend"
    / "modelA"
    / "analysis"
    / "detection_audit"
    / "threshold_test_results.csv"
)


# ==================================================
# THRESHOLDS
# ==================================================

THRESHOLDS = [
    0.50,
    0.30,
    0.20,
]


# ==================================================
# DETECTOR
# ==================================================

def create_detector(threshold):

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=threshold,
        min_hand_presence_confidence=threshold,
        min_tracking_confidence=0.5,
    )

    return (
        vision.HandLandmarker
        .create_from_options(options)
    )


# ==================================================
# MAIN
# ==================================================

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model asset missing:\n{MODEL_PATH}"
        )

    if not FAILED_CSV.exists():
        raise FileNotFoundError(
            f"Failed image report missing:\n{FAILED_CSV}"
        )


    # --------------------------------------------------
    # LOAD FAILED IMAGE LIST
    # --------------------------------------------------

    with open(
        FAILED_CSV,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        failed_images = list(
            csv.DictReader(file)
        )


    print("=" * 65)
    print("HANDLEX - DETECTOR THRESHOLD SENSITIVITY TEST")
    print("=" * 65)

    print(
        f"Images: {len(failed_images)}"
    )

    print(
        f"Thresholds: {THRESHOLDS}"
    )

    print()


    all_results = []


    # --------------------------------------------------
    # TEST EACH THRESHOLD
    # --------------------------------------------------

    for threshold in THRESHOLDS:

        print()
        print("-" * 65)

        print(
            f"TESTING THRESHOLD: {threshold}"
        )

        print("-" * 65)


        detector = create_detector(
            threshold
        )

        detected_count = 0


        try:

            for row in failed_images:

                filename = row["image"]

                image_path = (
                    TEST_DIR / filename
                )

                image = (
                    mp.Image.create_from_file(
                        str(image_path)
                    )
                )

                result = detector.detect(
                    image
                )

                detected = bool(
                    result.hand_landmarks
                )

                handedness = ""

                if detected:

                    detected_count += 1

                    if result.handedness:

                        handedness = (
                            result.handedness[0][0]
                            .category_name
                        )


                print(
                    f"{filename:20} | "
                    f"{'DETECTED' if detected else 'FAILED'}"
                    + (
                        f" | {handedness}"
                        if handedness
                        else ""
                    )
                )


                all_results.append({
                    "image": filename,
                    "expected": row["expected"],
                    "threshold": threshold,
                    "detected": detected,
                    "handedness": handedness,
                })


        finally:

            detector.close()


        print()

        print(
            f"Detection count: "
            f"{detected_count}/{len(failed_images)}"
        )

        print(
            f"Detection rate: "
            f"{detected_count / len(failed_images) * 100:.2f}%"
        )


    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "expected",
                "threshold",
                "detected",
                "handedness",
            ]
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )


    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print()
    print("=" * 65)
    print("FINAL THRESHOLD COMPARISON")
    print("=" * 65)


    for threshold in THRESHOLDS:

        subset = [
            row
            for row in all_results
            if row["threshold"] == threshold
        ]

        detected_count = sum(
            row["detected"]
            for row in subset
        )

        total = len(subset)

        rate = (
            detected_count / total * 100
            if total
            else 0
        )


        print(
            f"{threshold:.2f} | "
            f"{detected_count}/{total} | "
            f"{rate:.2f}%"
        )


    print()

    print(
        f"Results saved:\n{OUTPUT_CSV}"
    )

    print()
    print("TEST COMPLETE.")


if __name__ == "__main__":
    main()
