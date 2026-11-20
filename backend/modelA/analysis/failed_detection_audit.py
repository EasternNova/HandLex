
"""
HandLex - Failed Hand Detection Visual Audit

Inspects images that the existing inference recheck
reported as hand_detected=False.

Does NOT retrain or modify Model A.
Does NOT modify original images.
"""

from pathlib import Path
import csv

import cv2
import numpy as np


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_CSV = (
    PROJECT_ROOT
    / "backend"
    / "modelA"
    / "analysis"
    / "asl_inference_recheck.csv"
)

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "ASL"
    / "asl_alphabet_test"
    / "asl_alphabet_test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "modelA"
    / "analysis"
    / "detection_audit"
)

CONTACT_SHEET = (
    OUTPUT_DIR
    / "failed_detection_contact_sheet.jpg"
)

FAILED_CSV = (
    OUTPUT_DIR
    / "failed_detection_images.csv"
)


# ==================================================
# SETTINGS
# ==================================================

THUMB_WIDTH = 300
THUMB_HEIGHT = 300
LABEL_HEIGHT = 55
COLUMNS = 4


# ==================================================
# MAIN
# ==================================================

def main():

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Results CSV not found:\n{RESULTS_CSV}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    failed_images = []

    # --------------------------------------------------
    # READ PREVIOUS INFERENCE RESULTS
    # --------------------------------------------------

    with open(
        RESULTS_CSV,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            detected = (
                row["hand_detected"]
                .strip()
                .lower()
            )

            if detected in ("false", "0"):

                failed_images.append(row)


    print("=" * 60)
    print("HANDLEX - FAILED HAND DETECTION AUDIT")
    print("=" * 60)

    print(
        f"Failed images: {len(failed_images)}"
    )

    if not failed_images:

        print("No failed detection images found.")
        return


    # --------------------------------------------------
    # LOAD FAILED IMAGES
    # --------------------------------------------------

    tiles = []
    audit_rows = []

    for row in failed_images:

        filename = row["image"]

        image_path = (
            TEST_DIR / filename
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"Could not read: {filename}"
            )

            continue


        original_height, original_width = (
            image.shape[:2]
        )

        # Preserve aspect ratio
        scale = min(
            THUMB_WIDTH / original_width,
            THUMB_HEIGHT / original_height
        )

        resized_width = int(
            original_width * scale
        )

        resized_height = int(
            original_height * scale
        )

        resized = cv2.resize(
            image,
            (resized_width, resized_height)
        )


        # Black canvas for consistent tile dimensions
        canvas = np.zeros(
            (
                THUMB_HEIGHT + LABEL_HEIGHT,
                THUMB_WIDTH,
                3
            ),
            dtype=np.uint8
        )


        # Center image
        x_offset = (
            THUMB_WIDTH - resized_width
        ) // 2

        y_offset = (
            THUMB_HEIGHT - resized_height
        ) // 2

        canvas[
            y_offset:y_offset + resized_height,
            x_offset:x_offset + resized_width
        ] = resized


        # Image labels
        expected = row["expected"]

        cv2.putText(
            canvas,
            f"Expected: {expected}",
            (10, THUMB_HEIGHT + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            canvas,
            f"{original_width}x{original_height}",
            (10, THUMB_HEIGHT + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1
        )


        tiles.append(canvas)


        audit_rows.append({
            "image": filename,
            "expected": expected,
            "width": original_width,
            "height": original_height,
            "channels": image.shape[2],
            "hand_detected": False
        })


        print(
            f"{filename:20} | "
            f"{original_width}x{original_height}"
        )


    # --------------------------------------------------
    # SAVE FAILED IMAGE REPORT
    # --------------------------------------------------

    with open(
        FAILED_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "expected",
                "width",
                "height",
                "channels",
                "hand_detected"
            ]
        )

        writer.writeheader()

        writer.writerows(
            audit_rows
        )


    # --------------------------------------------------
    # CREATE CONTACT SHEET
    # --------------------------------------------------

    if not tiles:

        print(
            "No readable failed images."
        )

        return


    rows = (
        len(tiles) + COLUMNS - 1
    ) // COLUMNS


    tile_height = (
        THUMB_HEIGHT + LABEL_HEIGHT
    )


    sheet = np.zeros(
        (
            rows * tile_height,
            COLUMNS * THUMB_WIDTH,
            3
        ),
        dtype=np.uint8
    )


    for index, tile in enumerate(tiles):

        row_index = index // COLUMNS
        col_index = index % COLUMNS

        y = row_index * tile_height
        x = col_index * THUMB_WIDTH

        sheet[
            y:y + tile_height,
            x:x + THUMB_WIDTH
        ] = tile


    cv2.imwrite(
        str(CONTACT_SHEET),
        sheet
    )


    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("AUDIT FILES CREATED")
    print("=" * 60)

    print(
        f"Contact sheet:\n{CONTACT_SHEET}"
    )

    print(
        f"\nImage report:\n{FAILED_CSV}"
    )

    print()
    print("AUDIT COMPLETE.")


if __name__ == "__main__":
    main()
