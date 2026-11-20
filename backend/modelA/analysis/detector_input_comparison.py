
"""
HandLex - Detector Input Path & Resolution Comparison

Compares:
1. MediaPipe direct file loading
2. OpenCV BGR -> RGB
3. OpenCV BGR -> RGB -> 2x resize

Uses the original 0.50 detector threshold.
Does not modify model artifacts or original images.
"""

from pathlib import Path
import csv

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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
    / "input_comparison.csv"
)


def create_detector():

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.HandLandmarker.create_from_options(
        options
    )


def detect_rgb(detector, rgb):

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(image)

    return bool(result.hand_landmarks)


def main():

    with open(
        FAILED_CSV,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        failed_images = list(
            csv.DictReader(file)
        )


    detector = create_detector()

    results = []

    try:

        print("=" * 65)
        print("HANDLEX - INPUT PATH COMPARISON")
        print("=" * 65)

        for row in failed_images:

            filename = row["image"]

            path = TEST_DIR / filename

            # Method 1: MediaPipe direct file loading
            mp_image = mp.Image.create_from_file(
                str(path)
            )

            direct_result = detector.detect(
                mp_image
            )

            direct_detected = bool(
                direct_result.hand_landmarks
            )


            # Method 2 and 3: OpenCV
            bgr = cv2.imread(
                str(path)
            )

            if bgr is None:
                print(
                    f"Unreadable: {filename}"
                )
                continue

            rgb = cv2.cvtColor(
                bgr,
                cv2.COLOR_BGR2RGB
            )

            cv_detected = detect_rgb(
                detector,
                rgb
            )


            # Method 3: 2x resize
            rgb_2x = cv2.resize(
                rgb,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_LINEAR
            )

            resize_detected = detect_rgb(
                detector,
                rgb_2x
            )


            results.append({
                "image": filename,
                "expected": row["expected"],
                "direct_file": direct_detected,
                "opencv_rgb": cv_detected,
                "opencv_rgb_2x": resize_detected,
            })


            print(
                f"{filename:20} | "
                f"File: {str(direct_detected):5} | "
                f"RGB: {str(cv_detected):5} | "
                f"RGB 2x: {str(resize_detected):5}"
            )


    finally:

        detector.close()


    # Save report
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
                "direct_file",
                "opencv_rgb",
                "opencv_rgb_2x",
            ]
        )

        writer.writeheader()
        writer.writerows(results)


    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)

    for method in (
        "direct_file",
        "opencv_rgb",
        "opencv_rgb_2x"
    ):

        detected_count = sum(
            row[method]
            for row in results
        )

        total = len(results)

        print(
            f"{method:20} | "
            f"{detected_count}/{total}"
        )


    print()
    print(f"Report:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
