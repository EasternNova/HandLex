from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from landmarker import (
    create_landmarker,
    detect_file,
    get_first_hand,
)
from features import make_feature_vector


# ============================================================
# PATHS
# ============================================================

MODEL_A_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = MODEL_A_DIR.parent
PROJECT_ROOT = BACKEND_ROOT.parent


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "ASL"
    / "asl_alphabet_train"
    / "asl_alphabet_train"
)


DEFAULT_MODEL = (
    BACKEND_ROOT
    / "models"
    / "pretrained"
    / "hand_landmarker.task"
)


DEFAULT_OUTPUT_DIR = (
    BACKEND_ROOT
    / "data"
    / "modelA"
)


# ============================================================
# DATA EXTRACTION
# ============================================================

def extract(
    version,
    limit_per_class,
    output_path,
):

    print("=" * 60)
    print("HandLex ModelA - Landmark Dataset Preparation")
    print("=" * 60)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nDataset:")
    print(DEFAULT_DATASET)

    print("\nMediaPipe model:")
    print(DEFAULT_MODEL)

    print("\nFeature version:", version)

    if limit_per_class:
        print(
            "Limit per class:",
            limit_per_class,
        )
    else:
        print(
            "Limit per class: ALL"
        )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not DEFAULT_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DEFAULT_DATASET}"
        )

    if not DEFAULT_MODEL.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found:\n"
            f"{DEFAULT_MODEL}"
        )

    model_size = DEFAULT_MODEL.stat().st_size

    print(
        f"\nModel size: "
        f"{model_size:,} bytes"
    )

    if model_size < 1_000_000:
        raise RuntimeError(
            "MediaPipe model file is "
            "suspiciously small."
        )

    # --------------------------------------------------------
    # Find classes
    # --------------------------------------------------------

    classes = sorted(
        p.name
        for p in DEFAULT_DATASET.iterdir()
        if p.is_dir()
    )

    if not classes:
        raise RuntimeError(
            "No class folders found in:\n"
            f"{DEFAULT_DATASET}"
        )

    print(
        "\nClasses:",
        len(classes),
    )
    print(classes)

    # --------------------------------------------------------
    # Create detector
    # --------------------------------------------------------

    print(
        "\nCreating MediaPipe "
        "Hand Landmarker..."
    )

    detector = create_landmarker(
        DEFAULT_MODEL
    )

    print(
        "Hand Landmarker: OK"
    )

    X = []
    y = []
    paths = []

    failures = 0
    no_hand = 0
    processed = 0

    start_time = time.time()

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    try:

        for class_index, class_name in enumerate(
            classes
        ):

            class_dir = (
                DEFAULT_DATASET
                / class_name
            )

            images = []

            for extension in (
                "*.jpg",
                "*.jpeg",
                "*.png",
            ):
                images.extend(
                    class_dir.glob(
                        extension
                    )
                )

            images = sorted(images)

            if limit_per_class is not None:
                images = images[
                    :limit_per_class
                ]

            class_success = 0

            print(
                f"\n[{class_index + 1}/"
                f"{len(classes)}] "
                f"{class_name}: "
                f"{len(images)} images"
            )

            for image_path in images:

                processed += 1

                try:

                    result = detect_file(
                        detector,
                        image_path,
                    )

                    landmarks, handedness = (
                        get_first_hand(
                            result
                        )
                    )

                    # ------------------------------------------------
                    # Special handling for "nothing"
                    # ------------------------------------------------

                    if class_name == "nothing":

                        if landmarks is None:
                            no_hand += 1

                        feature_vector = (
                            make_feature_vector(
                                landmarks,
                                handedness,
                                version,
                            )
                        )

                        X.append(
                            feature_vector
                        )

                        y.append(
                            class_name
                        )

                        paths.append(
                            str(image_path)
                        )

                        class_success += 1

                        continue

                    # ------------------------------------------------
                    # Normal hand classes
                    # ------------------------------------------------

                    if landmarks is None:
                        failures += 1
                        continue

                    feature_vector = (
                        make_feature_vector(
                            landmarks,
                            handedness,
                            version,
                        )
                    )

                    X.append(
                        feature_vector
                    )

                    y.append(
                        class_name
                    )

                    paths.append(
                        str(image_path)
                    )

                    class_success += 1

                except Exception as e:

                    failures += 1

                    print(
                        f"  ERROR: "
                        f"{image_path.name} "
                        f"-> "
                        f"{type(e).__name__}: "
                        f"{e}"
                    )

            print(
                f"  Successful: "
                f"{class_success}/"
                f"{len(images)}"
            )

    finally:

        detector.close()

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    if not X:
        raise RuntimeError(
            "No feature vectors were created."
        )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(y)

    paths = np.asarray(paths)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = Path(
        output_path
    )

    if not output_path.is_absolute():
        output_path = (
            DEFAULT_OUTPUT_DIR
            / output_path
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        paths=paths,
        classes=np.asarray(
            classes
        ),
        version=np.asarray(
            version
        ),
    )

    elapsed = (
        time.time()
        - start_time
    )

    metadata = {
        "feature_version": version,
        "classes": classes,
        "num_classes": len(classes),
        "processed_images": processed,
        "successful_samples": len(X),
        "failures": failures,
        "no_hand_samples": no_hand,
        "feature_shape": list(
            X.shape
        ),
        "elapsed_seconds": elapsed,
        "dataset": str(
            DEFAULT_DATASET
        ),
        "model": str(
            DEFAULT_MODEL
        ),
    }

    metadata_path = (
        output_path.with_suffix(
            ".json"
        )
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )
    print(
        "EXTRACTION COMPLETE"
    )
    print(
        "=" * 60
    )

    print(
        "Processed images:",
        processed,
    )

    print(
        "Successful samples:",
        len(X),
    )

    print(
        "Failures:",
        failures,
    )

    print(
        "No-hand samples:",
        no_hand,
    )

    print(
        "Feature matrix shape:",
        X.shape,
    )

    print(
        "Labels shape:",
        y.shape,
    )

    print(
        "Classes:",
        len(classes),
    )

    print(
        "\nDataset saved to:"
    )
    print(output_path)

    print(
        "\nMetadata saved to:"
    )
    print(metadata_path)

    print(
        f"\nTime: "
        f"{elapsed:.2f} seconds"
    )

    print(
        "=" * 60
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Prepare MediaPipe landmark "
            "dataset for HandLex ModelA"
        )
    )

    parser.add_argument(
        "--version",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help=(
            "Feature version: "
            "1, 2, or 3"
        ),
    )

    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help=(
            "Maximum number of images "
            "per class"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npz path",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = (
            DEFAULT_OUTPUT_DIR
            / f"modelA_v{args.version}.npz"
        )

    extract(
        args.version,
        args.limit_per_class,
        args.output,
    )