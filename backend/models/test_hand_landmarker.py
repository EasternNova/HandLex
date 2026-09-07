from pathlib import Path
import sys

import mediapipe as mp


# ============================================================
# PATHS
# ============================================================

# This file is located at:
# G:\EasternNova\HandLex\backend\models\test_hand_landmarker.py

MODELS_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = MODELS_DIR.parent
PROJECT_ROOT = BACKEND_ROOT.parent

# MediaPipe model
MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

# ASL dataset
DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "ASL"
    / "asl_alphabet_train"
    / "asl_alphabet_train"
)

# New ModelA source code
MODEL_A_DIR = (
    BACKEND_ROOT
    / "src"
    / "models"
    / "modelA"
)

# Allow Python to import landmarker.py from ModelA
sys.path.insert(0, str(MODEL_A_DIR))

from landmarker import (
    create_landmarker,
    detect_file,
    get_first_hand,
)


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("HandLex ModelA - MediaPipe Hand Landmarker Test")
    print("=" * 60)

    # --------------------------------------------------------
    # Print paths
    # --------------------------------------------------------

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nBackend root:")
    print(BACKEND_ROOT)

    print("\nModel:")
    print(MODEL_PATH)

    print("\nDataset:")
    print(DATASET_ROOT)

    # --------------------------------------------------------
    # Check MediaPipe model
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("1. Checking MediaPipe model")
    print("-" * 60)

    if not MODEL_PATH.exists():
        print("RESULT: FAIL")
        print("MediaPipe model was not found:")
        print(MODEL_PATH)
        return

    model_size = MODEL_PATH.stat().st_size

    print("Model exists: OK")
    print(f"Model size: {model_size:,} bytes")

    if model_size < 1_000_000:
        print("RESULT: FAIL")
        print("Model file is suspiciously small.")
        return

    print("Model size: OK")

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("2. Checking ASL dataset")
    print("-" * 60)

    if not DATASET_ROOT.exists():
        print("RESULT: FAIL")
        print("Dataset folder was not found:")
        print(DATASET_ROOT)
        return

    print("Dataset root: OK")

    # We use class A for the first test
    class_a = DATASET_ROOT / "A"

    if not class_a.exists():
        print("RESULT: FAIL")
        print("Class A folder was not found:")
        print(class_a)
        return

    print("Class A folder: OK")

    # --------------------------------------------------------
    # Find an image
    # --------------------------------------------------------

    candidates = []

    candidates.extend(sorted(class_a.glob("*.jpg")))
    candidates.extend(sorted(class_a.glob("*.jpeg")))
    candidates.extend(sorted(class_a.glob("*.png")))

    if not candidates:
        print("RESULT: FAIL")
        print("No image found inside:")
        print(class_a)
        return

    image_path = candidates[0]

    print(f"Test image: {image_path}")
    print("Image: OK")

    # --------------------------------------------------------
    # Create MediaPipe detector
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("3. Creating MediaPipe Hand Landmarker")
    print("-" * 60)

    try:
        detector = create_landmarker(MODEL_PATH)
    except Exception as e:
        print("RESULT: FAIL")
        print("Could not create Hand Landmarker.")
        print(type(e).__name__ + ":", e)
        return

    print("Hand Landmarker: OK")

    # --------------------------------------------------------
    # Run detection
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("4. Running hand detection")
    print("-" * 60)

    try:
        result = detect_file(detector, image_path)

        hands_detected = len(result.hand_landmarks)

        print("Hands detected:", hands_detected)

        landmarks, handedness = get_first_hand(result)

    except Exception as e:
        print("RESULT: FAIL")
        print("Hand detection failed.")
        print(type(e).__name__ + ":", e)
        detector.close()
        return

    finally:
        try:
            detector.close()
        except Exception:
            pass

    # --------------------------------------------------------
    # Check landmarks
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("5. Checking landmarks")
    print("-" * 60)

    if landmarks is None:
        print("No hand was detected in the test image.")
        print()
        print("RESULT: FAIL - detector ran, but no hand was detected.")
        return

    landmark_count = len(landmarks)

    print("Landmarks detected:", landmark_count)

    if handedness:
        print("Handedness:", handedness)
    else:
        print("Handedness: unavailable")

    # --------------------------------------------------------
    # Print first 3 landmarks
    # --------------------------------------------------------

    print("\nFirst 3 landmarks:")

    for i, point in enumerate(landmarks[:3]):
        print(
            f"  {i}: "
            f"x={point.x:.5f}, "
            f"y={point.y:.5f}, "
            f"z={point.z:.5f}"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    if landmark_count == 21:
        print("RESULT: PASS - 21 landmarks available.")
    else:
        print(
            f"RESULT: FAIL - expected 21 landmarks, "
            f"got {landmark_count}."
        )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()