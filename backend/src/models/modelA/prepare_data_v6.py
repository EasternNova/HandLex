from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from landmarker import create_landmarker, detect_file, get_first_hand
from features import make_feature_vector


# ============================================================
# HandLex ModelA V6
# Full Dataset Landmark Extraction
#
# V6 improvements:
# - Uses V3's 95-feature representation
# - Processes the complete ASL training dataset
# - Handles "nothing" as a no-hand class
# - Saves checkpoints every 100 images
# - Automatically resumes after Ctrl+C / interruption
# - Final NPZ + JSON metadata are created only after
#   all classes have been processed
# ============================================================


# ============================================================
# PATHS
# ============================================================

MODEL_A_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = MODEL_A_DIR.parents[2]
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
    / "hand_landmarker.task"
)


DEFAULT_OUTPUT = (
    MODEL_A_DIR
    / "data"
    / "modelA_v6_full.npz"
)


# Checkpoints are stored separately from the final dataset.
DEFAULT_CHECKPOINT_DIR = (
    MODEL_A_DIR
    / "data"
    / "v6_checkpoints"
)


# ============================================================
# CONFIGURATION
# ============================================================

# V6 uses the already-developed V3 feature representation.
FEATURE_VERSION = 3

# V3 representation:
# 63 normalized XYZ coordinates
# + 10 finger joint angles
# + 20 wrist distances
# + 1 hand-presence feature
# + 1 handedness feature
# = 95 features
EXPECTED_FEATURES = 95


# Number of images processed before writing a checkpoint.
CHECKPOINT_SIZE = 100


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# ============================================================
# DISPLAY
# ============================================================

def print_header():
    print("=" * 70)
    print("HandLex ModelA V6 - Full Dataset Preparation")
    print("=" * 70)
    print()
    print("Dataset version      : V6")
    print("Feature representation: V3")
    print(f"Feature version      : {FEATURE_VERSION}")
    print(f"Expected features    : {EXPECTED_FEATURES}")
    print(f"Checkpoint size      : {CHECKPOINT_SIZE} images")
    print()
    print("This extractor is RESUMABLE.")
    print("Existing checkpoints will be skipped.")
    print("=" * 70)
    print()


# ============================================================
# DATASET HELPERS
# ============================================================

def find_classes(dataset_root: Path):
    """
    Find all class folders.

    The ASL dataset should contain exactly 29 classes:
        A-Z
        del
        nothing
        space
    """

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist:\n{dataset_root}"
        )

    classes = sorted(
        p.name
        for p in dataset_root.iterdir()
        if p.is_dir()
    )

    if len(classes) != 29:
        raise RuntimeError(
            f"Expected 29 class folders, found {len(classes)}:\n"
            f"{classes}"
        )

    return classes


def image_files(class_dir: Path):
    """
    Return all supported image files in deterministic order.
    """

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Class directory does not exist:\n{class_dir}"
        )

    return sorted(
        p
        for p in class_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# ============================================================
# FEATURE HELPERS
# ============================================================

def no_hand_features():
    """
    V3 representation for a frame/image where no hand exists.

    This is primarily used for the 'nothing' class.
    """

    return np.zeros(
        EXPECTED_FEATURES,
        dtype=np.float32,
    )


def validate_features(features):
    """
    Convert and validate one feature vector.
    """

    features = np.asarray(
        features,
        dtype=np.float32,
    ).reshape(-1)

    if features.shape[0] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"got {features.shape[0]}"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Feature vector contains NaN or Inf"
        )

    return features


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def checkpoint_path(
    checkpoint_dir: Path,
    class_index: int,
    start_index: int,
    end_index: int,
):
    """
    Generate a deterministic checkpoint filename.

    Example:
        class_00_00000_00100.npz
    """

    return (
        checkpoint_dir
        / (
            f"class_{class_index:02d}_"
            f"{start_index:05d}_"
            f"{end_index:05d}.npz"
        )
    )


def save_checkpoint(
    checkpoint_file: Path,
    X,
    y,
    paths,
    class_name,
    class_index,
    start_index,
    end_index,
    processed,
    successful,
    failures,
    no_hand_samples,
):
    """
    Save one checkpoint chunk.

    Every checkpoint is independent, so if the program is
    interrupted, already-completed chunks remain safe.
    """

    checkpoint_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if len(X) == 0:
        X_array = np.empty(
            (0, EXPECTED_FEATURES),
            dtype=np.float32
        )
    else:
        X_array = np.asarray(
            X,
            dtype=np.float32
        )

    if len(X_array) > 0:
        if X_array.ndim != 2:
            raise ValueError(
                f"Checkpoint X must be 2D, got {X_array.shape}"
            )

        if X_array.shape[1] != EXPECTED_FEATURES:
            raise ValueError(
                f"Checkpoint has {X_array.shape[1]} features; "
                f"expected {EXPECTED_FEATURES}"
            )

    y_array = np.asarray(
        y,
        dtype=np.int64,
    )

    paths_array = np.asarray(
        paths,
        dtype=str,
    )

    metadata = {
        "class_name": class_name,
        "class_index": class_index,
        "start_index": start_index,
        "end_index": end_index,
        "processed": processed,
        "successful": successful,
        "failures": failures,
        "no_hand_samples": no_hand_samples,
        "feature_version": FEATURE_VERSION,
        "feature_count": EXPECTED_FEATURES,
    }

    np.savez_compressed(
        checkpoint_file,
        X=X_array,
        y=y_array,
        paths=paths_array,
        metadata=np.asarray(
            json.dumps(metadata)
        ),
    )


def load_checkpoint(checkpoint_file: Path):
    """
    Load one checkpoint.
    """

    data = np.load(
        checkpoint_file,
        allow_pickle=False,
    )

    X = data["X"]
    y = data["y"]
    paths = data["paths"]

    metadata = json.loads(
        data["metadata"].item()
    )

    return X, y, paths, metadata


def list_checkpoints(checkpoint_dir: Path):
    """
    Return all checkpoint files in deterministic order.
    """

    if not checkpoint_dir.exists():
        return []

    return sorted(
        checkpoint_dir.glob("class_*.npz")
    )


# ============================================================
# CHECKPOINT SUMMARY
# ============================================================

def print_existing_checkpoints(checkpoint_dir: Path):
    """
    Display checkpoints already available before extraction.
    """

    checkpoints = list_checkpoints(
        checkpoint_dir
    )

    if not checkpoints:
        print("No existing V6 checkpoints found.")
        print("Extraction will start from the beginning.")
        print()
        return

    print(
        f"Found {len(checkpoints)} existing checkpoint(s)."
    )
    print(
        "Previously completed chunks will be skipped."
    )
    print()

    for checkpoint in checkpoints[:10]:
        print(
            f"  {checkpoint.name}"
        )

    if len(checkpoints) > 10:
        print(
            f"  ... and {len(checkpoints) - 10} more"
        )

    print()


# ============================================================
# MERGE CHECKPOINTS
# ============================================================

def merge_checkpoints(
    checkpoint_dir: Path,
    output_path: Path,
    classes,
    dataset_root: Path,
    elapsed_seconds: float,
):
    """
    Merge all checkpoint chunks into the final V6 dataset.
    """

    checkpoints = list_checkpoints(
        checkpoint_dir
    )

    if not checkpoints:
        raise RuntimeError(
            "No checkpoints were found. "
            "Nothing can be merged."
        )

    print()
    print("=" * 70)
    print("MERGING V6 CHECKPOINTS")
    print("=" * 70)
    print()
    print(
        f"Checkpoint files: {len(checkpoints):,}"
    )

    all_X = []
    all_y = []
    all_paths = []

    per_class = {
        class_name: {
            "images": 0,
            "successful": 0,
            "failures": 0,
            "no_hand": 0,
        }
        for class_name in classes
    }

    total_images = 0
    successful = 0
    failures = 0
    no_hand_samples = 0

    for checkpoint_file in checkpoints:

        X, y, paths, metadata = load_checkpoint(
            checkpoint_file
        )

        # Validate checkpoint feature shape.
        if X.ndim != 2:
            raise RuntimeError(
                f"Invalid checkpoint shape in "
                f"{checkpoint_file}: {X.shape}"
            )

        if X.shape[1] != EXPECTED_FEATURES:
            raise RuntimeError(
                f"Invalid feature count in "
                f"{checkpoint_file}: "
                f"{X.shape[1]}"
            )

        if not np.isfinite(X).all():
            raise RuntimeError(
                f"NaN/Inf found in checkpoint: "
                f"{checkpoint_file}"
            )

        all_X.append(X)
        all_y.append(y)
        all_paths.append(paths)

        class_name = metadata["class_name"]

        if class_name not in per_class:
            raise RuntimeError(
                f"Unknown class in checkpoint: "
                f"{class_name}"
            )

        per_class[class_name]["images"] += metadata[
            "processed"
        ]

        per_class[class_name]["successful"] += metadata[
            "successful"
        ]

        per_class[class_name]["failures"] += metadata[
            "failures"
        ]

        per_class[class_name]["no_hand"] += metadata[
            "no_hand_samples"
        ]

        total_images += metadata["processed"]
        successful += metadata["successful"]
        failures += metadata["failures"]
        no_hand_samples += metadata["no_hand_samples"]

    # --------------------------------------------------------
    # Final arrays
    # --------------------------------------------------------

    X_array = np.concatenate(
        all_X,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    y_array = np.concatenate(
        all_y,
        axis=0,
    ).astype(
        np.int64,
        copy=False,
    )

    paths_array = np.concatenate(
        all_paths,
        axis=0,
    )

    classes_array = np.asarray(
        classes,
        dtype=str,
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if X_array.shape[0] != y_array.shape[0]:
        raise RuntimeError(
            "X/y sample count mismatch: "
            f"{X_array.shape[0]} vs {y_array.shape[0]}"
        )

    if X_array.shape[0] != paths_array.shape[0]:
        raise RuntimeError(
            "X/paths sample count mismatch: "
            f"{X_array.shape[0]} vs "
            f"{paths_array.shape[0]}"
        )

    if X_array.shape[1] != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Final feature count is "
            f"{X_array.shape[1]}, expected "
            f"{EXPECTED_FEATURES}"
        )

    if not np.isfinite(X_array).all():
        raise RuntimeError(
            "Final feature matrix contains NaN or Inf."
        )

    # --------------------------------------------------------
    # Detection rate
    # --------------------------------------------------------

    detection_rate = (
        successful / total_images
        if total_images
        else 0.0
    )

    # --------------------------------------------------------
    # Save final NPZ
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        X=X_array,
        y=y_array,
        paths=paths_array,
        classes=classes_array,
        version=np.asarray(
            FEATURE_VERSION
        ),
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_path = output_path.with_suffix(
        ".json"
    )

    metadata = {
        "model": "HandLex ModelA",
        "dataset_version": "V6",
        "feature_representation": "V3",
        "feature_version": FEATURE_VERSION,
        "feature_count": EXPECTED_FEATURES,

        "dataset_root": str(
            dataset_root
        ),

        "total_images": total_images,
        "successful_samples": successful,
        "failures": failures,
        "no_hand_samples": no_hand_samples,

        "detection_rate": detection_rate,

        "classes": classes,
        "num_classes": len(classes),

        "feature_shape": list(
            X_array.shape
        ),

        "label_shape": list(
            y_array.shape
        ),

        "per_class": per_class,

        "checkpoint_count": len(
            checkpoints
        ),

        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": (
            elapsed_seconds / 60.0
        ),
    }

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

    print()
    print("=" * 70)
    print("V6 EXTRACTION COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Images processed : {total_images:,}"
    )
    print(
        f"Usable samples   : {successful:,}"
    )
    print(
        f"Failures         : {failures:,}"
    )
    print(
        f"No-hand samples  : {no_hand_samples:,}"
    )
    print(
        f"Detection rate   : "
        f"{detection_rate * 100:.2f}%"
    )
    print(
        f"Feature matrix   : "
        f"{X_array.shape}"
    )
    print(
        f"Labels           : "
        f"{y_array.shape}"
    )
    print(
        f"Classes          : "
        f"{len(classes)}"
    )
    print(
        f"Finite features  : "
        f"{np.isfinite(X_array).all()}"
    )
    print()

    # --------------------------------------------------------
    # Per-class summary
    # --------------------------------------------------------

    print("Per-class summary:")
    print()

    print(
        f"{'CLASS':<10}"
        f"{'IMAGES':>10}"
        f"{'USABLE':>10}"
        f"{'FAILED':>10}"
        f"{'RATE':>10}"
    )

    print("-" * 50)

    for class_name in classes:

        info = per_class[class_name]

        rate = (
            info["successful"]
            / info["images"]
            if info["images"]
            else 0.0
        )

        print(
            f"{class_name:<10}"
            f"{info['images']:>10,}"
            f"{info['successful']:>10,}"
            f"{info['failures']:>10,}"
            f"{rate * 100:>9.2f}%"
        )

    print()

    print("=" * 70)
    print("V6 DATASET SAVED")
    print("=" * 70)
    print()

    print(
        f"NPZ     : {output_path}"
    )

    print(
        f"Metadata: {metadata_path}"
    )

    print()

    print(
        "Next step: validate the V6 NPZ before training."
    )

    print()


# ============================================================
# MAIN EXTRACTION
# ============================================================

def main(
    dataset_root: Path,
    model_path: Path,
    output_path: Path,
    checkpoint_dir: Path,
    limit_per_class: int | None,
):
    print_header()

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------

    dataset_root = dataset_root.resolve()
    model_path = model_path.resolve()
    output_path = output_path.resolve()
    checkpoint_dir = checkpoint_dir.resolve()

    # --------------------------------------------------------
    # Safety: do not overwrite final dataset
    # --------------------------------------------------------

    if output_path.exists():
        raise FileExistsError(
            f"Final V6 dataset already exists: "
            f"{output_path}\n"
            "Remove it only if you intentionally want "
            "to rebuild the complete dataset."
        )

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_root}"
        )

    # --------------------------------------------------------
    # Validate MediaPipe model
    # --------------------------------------------------------

    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found:\n{model_path}"
        )

    model_size = model_path.stat().st_size

    if model_size < 1_000_000:
        raise RuntimeError(
            "MediaPipe model file is suspiciously small.\n"
            f"Path: {model_path}\n"
            f"Size: {model_size:,} bytes"
        )

    # --------------------------------------------------------
    # Print paths
    # --------------------------------------------------------

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Dataset      : {dataset_root}"
    )

    print(
        f"MediaPipe    : {model_path}"
    )

    print(
        f"Model size   : {model_size:,} bytes"
    )

    print(
        f"Output       : {output_path}"
    )

    print(
        f"Checkpoints  : {checkpoint_dir}"
    )

    print()

    # --------------------------------------------------------
    # Find classes
    # --------------------------------------------------------

    classes = find_classes(
        dataset_root
    )

    print(
        f"Classes ({len(classes)}):"
    )

    print(classes)
    print()

    # --------------------------------------------------------
    # Prepare directories
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Show existing checkpoints
    # --------------------------------------------------------

    print_existing_checkpoints(
        checkpoint_dir
    )

    # --------------------------------------------------------
    # Create MediaPipe detector
    # --------------------------------------------------------

    print(
        "Creating MediaPipe Hand Landmarker..."
    )

    detector = create_landmarker(
        model_path
    )

    print(
        "Hand Landmarker: OK"
    )

    print()

    start_time = time.time()

    # ========================================================
    # EXTRACTION
    # ========================================================

    try:

        for class_index, class_name in enumerate(
            classes
        ):

            class_dir = (
                dataset_root
                / class_name
            )

            files = image_files(
                class_dir
            )

            # ------------------------------------------------
            # Optional smoke-test limit
            # ------------------------------------------------

            if limit_per_class is not None:
                files = files[
                    :limit_per_class
                ]

            class_total = len(files)

            print("=" * 70)

            print(
                f"[{class_index + 1:02d}/"
                f"{len(classes):02d}] "
                f"{class_name}: "
                f"{class_total:,} images"
            )

            print("=" * 70)

            # ------------------------------------------------
            # Process in chunks
            # ------------------------------------------------

            for chunk_start in range(
                0,
                class_total,
                CHECKPOINT_SIZE,
            ):

                chunk_end = min(
                    chunk_start
                    + CHECKPOINT_SIZE,
                    class_total,
                )

                checkpoint_file = checkpoint_path(
                    checkpoint_dir,
                    class_index,
                    chunk_start,
                    chunk_end,
                )

                # --------------------------------------------
                # RESUME LOGIC
                # --------------------------------------------

                if checkpoint_file.exists():

                    print(
                        f"  SKIP checkpoint "
                        f"{chunk_start:,}-"
                        f"{chunk_end:,}: "
                        f"{checkpoint_file.name}"
                    )

                    continue

                print(
                    f"  Processing "
                    f"{chunk_start + 1:,}-"
                    f"{chunk_end:,}..."
                )

                # --------------------------------------------
                # Chunk containers
                # --------------------------------------------

                chunk_X = []
                chunk_y = []
                chunk_paths = []

                chunk_processed = 0
                chunk_successful = 0
                chunk_failures = 0
                chunk_no_hand = 0

                # --------------------------------------------
                # Process images
                # --------------------------------------------

                for image_index in range(
                    chunk_start,
                    chunk_end,
                ):

                    image_path = files[
                        image_index
                    ]

                    chunk_processed += 1

                    try:

                        # ------------------------------------
                        # MediaPipe detection
                        # ------------------------------------

                        result = detect_file(
                            detector,
                            image_path,
                        )

                        landmarks, handedness = (
                            get_first_hand(
                                result
                            )
                        )

                        # ------------------------------------
                        # NOTHING CLASS
                        # ------------------------------------
                        #
                        # "nothing" is special.
                        #
                        # If no hand is detected:
                        #     use a zero vector.
                        #
                        # If a hand is detected:
                        #     use its actual landmarks.
                        #
                        # ------------------------------------

                        if class_name == "nothing":

                            if landmarks is None:

                                features = (
                                    no_hand_features()
                                )

                                chunk_no_hand += 1

                            else:

                                features = (
                                    make_feature_vector(
                                        landmarks,
                                        handedness,
                                        FEATURE_VERSION,
                                    )
                                )

                        # ------------------------------------
                        # NORMAL CLASSES
                        # ------------------------------------

                        else:

                            if landmarks is None:

                                chunk_failures += 1

                                print(
                                    f"    NO HAND: "
                                    f"{image_path.name}"
                                )

                                continue

                            features = (
                                make_feature_vector(
                                    landmarks,
                                    handedness,
                                    FEATURE_VERSION,
                                )
                            )

                        # ------------------------------------
                        # Validate features
                        # ------------------------------------

                        features = validate_features(
                            features
                        )

                        # ------------------------------------
                        # Store
                        # ------------------------------------

                        chunk_X.append(
                            features
                        )

                        chunk_y.append(
                            class_index
                        )

                        chunk_paths.append(
                            str(image_path)
                        )

                        chunk_successful += 1

                    except Exception as exc:

                        chunk_failures += 1

                        print(
                            f"    ERROR: "
                            f"{image_path.name} -> "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )

                # --------------------------------------------
                # Save checkpoint
                # --------------------------------------------

                save_checkpoint(
                    checkpoint_file,
                    chunk_X,
                    chunk_y,
                    chunk_paths,
                    class_name,
                    class_index,
                    chunk_start,
                    chunk_end,
                    chunk_processed,
                    chunk_successful,
                    chunk_failures,
                    chunk_no_hand,
                )

                print(
                    f"    CHECKPOINT SAVED"
                )

                print(
                    f"    range    : "
                    f"{chunk_start + 1:,}-"
                    f"{chunk_end:,}"
                )

                print(
                    f"    processed: "
                    f"{chunk_processed:,}"
                )

                print(
                    f"    usable   : "
                    f"{chunk_successful:,}"
                )

                print(
                    f"    failed   : "
                    f"{chunk_failures:,}"
                )

                print(
                    f"    no-hand  : "
                    f"{chunk_no_hand:,}"
                )

                print()

            # ------------------------------------------------
            # Class complete
            # ------------------------------------------------

            class_checkpoints = [
                p
                for p in list_checkpoints(
                    checkpoint_dir
                )
                if f"class_{class_index:02d}_"
                in p.name
            ]

            class_images = 0
            class_successful = 0
            class_failures = 0
            class_no_hand = 0

            for checkpoint_file in (
                class_checkpoints
            ):

                _, _, _, metadata = (
                    load_checkpoint(
                        checkpoint_file
                    )
                )

                class_images += metadata[
                    "processed"
                ]

                class_successful += metadata[
                    "successful"
                ]

                class_failures += metadata[
                    "failures"
                ]

                class_no_hand += metadata[
                    "no_hand_samples"
                ]

            class_rate = (
                class_successful
                / class_images
                if class_images
                else 0.0
            )

            print(
                f"  CLASS COMPLETE: "
                f"{class_name}"
            )

            print(
                f"    images   = "
                f"{class_images:,}"
            )

            print(
                f"    usable   = "
                f"{class_successful:,}"
            )

            print(
                f"    failed   = "
                f"{class_failures:,}"
            )

            print(
                f"    no-hand  = "
                f"{class_no_hand:,}"
            )

            print(
                f"    rate     = "
                f"{class_rate * 100:.2f}%"
            )

            print()

    finally:

        detector.close()

    # ========================================================
    # MERGE EVERYTHING
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    merge_checkpoints(
        checkpoint_dir,
        output_path,
        classes,
        dataset_root,
        elapsed,
    )


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Prepare the complete HandLex "
            "ModelA V6 landmark dataset "
            "with resumable checkpoints."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "ASL training dataset root"
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=(
            "MediaPipe hand_landmarker.task path"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Final V6 NPZ output path"
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=(
            "Directory used for resumable "
            "V6 checkpoints"
        ),
    )

    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help=(
            "Optional number of images per class "
            "for a smoke test. "
            "Omit for the complete dataset."
        ),
    )

    args = parser.parse_args()

    main(
        dataset_root=args.dataset,
        model_path=args.model,
        output_path=args.output,
        checkpoint_dir=args.checkpoint_dir,
        limit_per_class=args.limit_per_class,
    )