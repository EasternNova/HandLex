
"""
HandLex - Model A V6 External ASL Dataset Recheck

Uses the existing inference engine.
Does NOT retrain or modify Model A.
"""

from pathlib import Path
import csv
import sys

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Import the existing inference engine
sys.path.insert(0, str(PROJECT_ROOT))

from backend.modelA.inference import ModelAInference


# ==================================================
# PATHS
# ==================================================

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "ASL"
    / "asl_alphabet_test"
    / "asl_alphabet_test"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "backend"
    / "modelA"
    / "analysis"
    / "asl_inference_recheck.csv"
)


# ==================================================
# VALIDATION
# ==================================================

def main():

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_DIR}"
        )

    image_files = sorted(
        TEST_DIR.glob("*_test.jpg")
    )

    if not image_files:
        raise FileNotFoundError(
            "No *_test.jpg images found."
        )

    print("=" * 60)
    print("HANDLEX - MODEL A V6 EXTERNAL INFERENCE RECHECK")
    print("=" * 60)

    print(f"Dataset: {TEST_DIR}")
    print(f"Images found: {len(image_files)}")
    print()

    engine = ModelAInference()

    results = []
    correct = 0
    hand_detected_count = 0
    failed_count = 0

    try:

        for image_path in image_files:

            # Filename:
            # A_test.jpg -> A
            # nothing_test.jpg -> nothing
            # space_test.jpg -> space

            expected = (
                image_path.stem
                .removesuffix("_test")
            )

            try:

                result = engine.predict_image(
                    image_path
                )

                predicted = result["prediction"]
                confidence = result["confidence"]
                hand_detected = result["hand_detected"]
                handedness = result["handedness"]

                is_correct = (
                    predicted.lower()
                    == expected.lower()
                )

                if is_correct:
                    correct += 1

                if hand_detected:
                    hand_detected_count += 1

                status = (
                    "CORRECT"
                    if is_correct
                    else "WRONG"
                )

                print(
                    f"{expected:10} | "
                    f"{predicted:10} | "
                    f"{confidence:.4f} | "
                    f"{status}"
                )

                results.append({
                    "image": image_path.name,
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": confidence,
                    "hand_detected": hand_detected,
                    "handedness": handedness,
                    "correct": is_correct,
                    "top3": result["top3"],
                })

            except Exception as error:

                failed_count += 1

                print(
                    f"{expected:10} | "
                    f"ERROR: {error}"
                )

                results.append({
                    "image": image_path.name,
                    "expected": expected,
                    "predicted": "ERROR",
                    "confidence": 0,
                    "hand_detected": False,
                    "handedness": None,
                    "correct": False,
                    "top3": [],
                })

    finally:

        engine.close()


    # ==================================================
    # SUMMARY
    # ==================================================

    total = len(image_files)
    successful = total - failed_count

    accuracy = (
        correct / successful * 100
        if successful
        else 0
    )

    detection_rate = (
        hand_detected_count / successful * 100
        if successful
        else 0
    )

    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Total images:          {total}")
    print(f"Successful inference:  {successful}")
    print(f"Failed inference:      {failed_count}")
    print(f"Correct predictions:  {correct}")
    print(f"Incorrect predictions: {successful - correct}")
    print(f"Accuracy:              {accuracy:.2f}%")
    print(f"Hand detected:         {hand_detected_count}")
    print(f"Detection rate:        {detection_rate:.2f}%")

    print()


    # ==================================================
    # SAVE CSV
    # ==================================================

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
                "predicted",
                "confidence",
                "hand_detected",
                "handedness",
                "correct",
                "top3",
            ]
        )

        writer.writeheader()

        for row in results:

            row_to_write = row.copy()

            row_to_write["top3"] = str(
                row_to_write["top3"]
            )

            writer.writerow(
                row_to_write
            )

    print(f"CSV saved to:\n{OUTPUT_CSV}")

    print()
    print("RECHECK COMPLETE.")


if __name__ == "__main__":
    main()
