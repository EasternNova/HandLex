# HandLex Model A — Historical Archive

This archive preserves retired Model A experiments and enough associated evidence to explain and reproduce them where the supplied files permit. It deliberately does **not** create one folder per version. Historical scripts are stored once under `scripts/`; version-specific datasets and model artifacts are under `evidence/`.

> **Safety note:** This archive was assembled from the supplied `modelA.zip` snapshot. The original working directory was not directly accessible or modified. Review the resulting package, then apply the same moves to `G:\EasternNova\HandLex\backend\modelA` only after making a backup.

## Quick version timeline

| Version | Approach evidenced in source | Evidence in this archive | Recorded result |
|---|---|---|---|
| V1 | Raw MediaPipe XYZ coordinates; 21 landmarks × 3 = 63 features. | `modelA_v1_50.npz`, `test_v1.npz`; shared preparation script. | No V1 trained-model metadata or accuracy report was present in the supplied snapshot. A prior small extraction test reported 32 samples, 29 classes and 63 features; that is a pipeline smoke test, not model accuracy. |
| V2 | Wrist-relative translation and scale normalization; 63 features. Random Forest. | V2 dataset, RF pickle and JSON metadata. | 1,290 samples; 1,032 train / 258 test; recorded test accuracy **98.8372%**. |
| V3 | V2 normalized XYZ + 10 joint angles + 20 wrist-to-landmark distances + hand-present + handedness = 95 features. Random Forest. | V3 dataset, RF pickle and JSON metadata. | 1,290 samples; 1,032 train / 258 test; recorded test accuracy **99.6124%**. |
| V4 | XGBoost classifier using the V3 95-feature representation and V3 dataset. | XGBoost pickle and JSON metadata. | 1,290 samples; 1,032 train / 258 test; recorded test accuracy **97.2868%**. |
| V5 | Webcam inference / live landmark visualization and temporal prediction stabilization. The supplied webcam code labels itself V5.1. This is an application/inference iteration, not evidence of a new trained feature architecture. | Historical `webcam_v5.py`; current `webcam.py` remains outside the archive. | No V5-specific evaluation metrics were supplied. |
| V6 (active) | Full-dataset landmark extraction using V3's 95-feature representation; `nothing` no-hand handling; resumable extraction checkpoints; Random Forest classifier. | **Kept in the active model directory**, not moved here. | 66,589 samples; 53,271 train / 13,318 test; recorded accuracy **98.8287%**. |
| V7 | V6 dataset samples and labels with hand-relative rotation canonicalization; V3-style 95-feature representation; Random Forest. | V7 scripts, canonicalized dataset, RF pickle and JSON metadata. | 66,589 samples; 53,271 train / 13,318 test; recorded accuracy **97.6573%**. |

Recorded metrics above are copied from the versioned JSON artifact metadata in the supplied snapshot. They have **not** been independently rerun as part of this archive operation. Accuracy comparisons are meaningful only with the stated split and evaluation setup.

## What is stored where

### `scripts/`

- `prepare_data.py` — earlier configurable V1/V2/V3 extraction pipeline. It accepts a feature-version argument; V1 uses raw XYZ, V2 uses normalized XYZ, V3 uses the 95-feature engineered representation.
- `webcam_v5.py` — historical V5.1 diagnostic webcam implementation. The root `webcam.py` is retained as the current webcam entry point; both files should be diffed before deleting either.
- `prepare_data_v7.py` — builds the V7 canonicalized dataset from the V6 dataset.
- `train_v7.py` — trains the V7 Random Forest and writes its model artifact and metadata.
- `analyze_v7.py` — V7 test-split confusion-matrix analysis.
- `validate_v7.py` — compares V6 and V7 dataset dimensions, labels and feature representations.

### `evidence/artifacts/`

Versioned trained-model artifacts and their JSON metadata:
- V2: `modelA_rf_v2.pkl` and `.json`
- V3: `modelA_rf_v3.pkl` and `.json`
- V4: `modelA_xgb_v4.pkl` and `.json`
- V7: `modelA_rf_v7.pkl` and `.json`

The JSON files are the human-readable model cards: classifier, feature version/count, class list, sample counts, split, accuracy and configuration. Keep the matching `.pkl` and `.json` together.

### `evidence/data/`

Small historical datasets and metadata:
- V1: `modelA_v1_50.npz` and `modelA_v1_50.json`
- V1 smoke test: `test_v1.npz` and `test_v1.json`
- V2: `modelA_v2_50.npz` and `modelA_v2_50.json`
- V3: `modelA_v3_50.npz` and `modelA_v3_50.json`
- V7: `modelA_v7_canonical.npz` and `modelA_v7_canonical.json`

These are compact extracted feature datasets, not the original image/video corpus.

## Active V6 files intentionally retained outside this archive

The following V6 implementation and audit files remain at the `modelA/` root:

- `prepare_data_v6.py` — V6 full-dataset extraction, chunk checkpoints, resume/merge logic.
- `analyze_v6.py` — V6 evaluation on the stratified test split.
- `audit_v6_nothing.py` — audit of the `nothing` class.
- `canonicalization_audit_v6.py` — V6 canonicalization investigation.
- `canonicalization_classification_v6.py` — classifier experiment using canonicalized features.
- `canonicalized_separability_v6.py` — class-separability analysis.
- `feature_importance_v6.py` — feature-importance analysis.
- `orientation_audit_v6.py` — orientation/geometry audit.
- `permutation_v6.py` — permutation experiment on the V6 test split.
- `rotation_sensitivity_v6.py` — rotation sensitivity experiment.
- `analyze_detection_failures.py` — reads V6 extraction metadata and summarizes detection failures.
- `analyze_nm_confidence.py`, `analyze_nm_errors.py`, `make_nm_sheet.py`, `nm_error_images.jpg` — N/M confusion investigation materials.
- `features.py` — shared feature construction used by the V6 pipeline and webcam.
- `landmarker.py` — MediaPipe hand landmark detection used by V6 extraction.
- `train.py` — general trainer supports feature versions 1–3 and is retained because V6 uses the 95-feature (feature version 3) representation.
- `webcam.py` — retained as the current webcam entry point; its source identifies it as V5.1 and it supports feature-version-aware inference.
- `model.py`, `test_hand_landmarker.py`, `__init__.py` — retained pending a complete external-import check.

V6 data and model evidence also remain outside the archive:
- `data/modelA_v6_full.npz` and `.json`
- `data/v6_test.npz` and `.json`
- `artifacts/modelA_rf_v6.pkl` and `.json`
- `checkpoints/v6_checkpoints/` (870 chunk files in the supplied snapshot)

The V6 checkpoint chunks are extraction-resume chunks, not independent trained model checkpoints. They are retained because they support recovery/reconstruction of the full extracted feature dataset.

## Findings and limitations — do not overstate

1. V2's recorded accuracy is 98.8372%; V3's is 99.6124%. The V3 feature representation adds geometric features to normalized XYZ.
2. V4's recorded XGBoost accuracy is 97.2868%, lower than the V3 Random Forest result on the recorded small dataset/split. This establishes a lower measured result in these metadata files; it does **not** prove why XGBoost performed worse.
3. V6's recorded full-dataset accuracy is 98.8287%.
4. V7 uses the same sample/label set and stated split/configuration as V6, but its recorded accuracy is 97.6573%. Thus V7 did not improve the recorded accuracy on this evaluation. The specific cause of the decrease is **not established** by the metadata alone; do not claim canonicalization itself is definitively the cause without inspecting audit outputs.
5. The V1 smoke-test evidence demonstrates feature extraction output shape and sample-level hand-detection behavior only. It is not a trained classifier result.
6. No V5 accuracy/validation report, V1 trained artifact, or complete V2–V5 experiment logs were present in the supplied ZIP. Those details remain unknown unless recovered from older logs or conversations.
7. `analyze_nm_confidence.py` currently points to `backend/models/modelA_rf_v6.pkl`, while the supplied V6 artifact is located at `backend/modelA/artifacts/modelA_rf_v6.pkl`. Treat this as a likely stale path that needs correction before running that script; it was not modified here.
8. `webcam.py` and `webcam_v5.py` are very similar but not byte-identical. Keep the root webcam file until its imports and runtime use are verified.

## Reproduction notes

Run historical scripts from the HandLex backend/project root only after checking their relative paths and installed dependencies. Historical scripts may expect the original dataset directories and a particular working directory. The archive contains extracted feature arrays and model artifacts, but not the complete original ASL source image dataset.

For each version, use the matching feature representation and artifact:
- V1: 63 raw XYZ features
- V2: 63 wrist-centered, scale-normalized XYZ features
- V3/V4/V6: 95 features (V3 engineered geometry; V4 uses XGBoost on V3 features; V6 uses Random Forest)
- V7: 95 canonicalized features

Do not load a model with a feature vector of a different version/width. Do not overwrite the V6 artifact while testing historical versions.

## Cleanup record

- Source snapshot: user-provided `modelA.zip`.
- Archive layout: flat historical scripts plus shared evidence subfolders; no per-version directories and no `shared/` directory.
- No original source files were deleted from the uploaded ZIP. Historical files were relocated only inside the generated cleaned copy.
- The original `G:\EasternNova\HandLex\backend\modelA` folder was not directly accessible from this environment. Apply the proposed structure to the local folder only after backing it up and verifying the active model.
