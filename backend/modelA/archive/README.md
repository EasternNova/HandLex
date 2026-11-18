# HandLex Model A — Historical Archive

## Purpose
This archive preserves the superseded Model A experiments and their reproducibility evidence.
The active Model A/V6 implementation remains in the parent `modelA/` directory.

## Important safety boundary
Do NOT move or delete the parent's V6 working files as part of installing this archive.
In the supplied snapshot, the following are retained outside this archive because they are
V6/current-path assets or shared by the active implementation:

- `features.py`, `landmarker.py`, `model.py`
- `prepare_data_v6.py`, `train.py`, `webcam.py`, `test_hand_landmarker.py`, `__init__.py`
- V6 analysis scripts (`analyze_v6.py`, `audit_v6_nothing.py`, canonicalization/orientation/
  permutation/rotation/feature-importance scripts, and the NM/detection analysis scripts)
- `data/modelA_v6_full.*`, `data/v6_test.*`
- `artifacts/modelA_rf_v6.*`
- `checkpoints/v6_checkpoints/` (resumable extraction chunks)
- `artifacts/` and `data/` folders remain in place; only the explicitly listed historical
  files are relocated into this archive when installed.

`webcam.py` is retained as the generic current webcam entry point. `webcam_v5.py` is
archived as a historical snapshot; it is not assumed to be byte-for-byte identical.

## Archive contents and version map

### V1 — initial baseline / extraction test
- `code/prepare_data.py`: earlier versioned feature-extraction pipeline. The script accepts
  a feature-version argument; inspect its CLI/defaults when reproducing.
- `data/modelA_v1_50.npz` and metadata: small V1 dataset.
- `data/test_v1.npz` and metadata: tiny pipeline test dataset.
- No V1 model artifact was present in the supplied `artifacts/` listing.
- Historical test evidence from the project conversation: 32 successful detections,
  26 failures, 2 no-hand samples from 58 sampled images; output feature shape (32, 63),
  29 classes, ~2.93 seconds. This is a small extraction test, not a full model benchmark.
- Failure interpretation: 26 images had no detected hand in that sample; the source evidence
  does not establish that the whole dataset or model failed.

### V2 — Random Forest, 63 features
- `artifacts/modelA_rf_v2.json/.pkl`
- `data/modelA_v2_50.npz` and metadata.
- Artifact metadata reports RandomForest, feature version 2, 63 features, 29 classes,
  1,290 samples (1,032 train / 258 test), random_state 42, test accuracy 0.988372
  (98.84%).
- The metadata establishes measured test accuracy, not real-world webcam accuracy.

### V3 — Random Forest, 95 features
- `artifacts/modelA_rf_v3.json/.pkl`
- `data/modelA_v3_50.npz` and metadata.
- Artifact metadata reports RandomForest, feature version 3, 95 features, 29 classes,
  1,290 samples (1,032 train / 258 test), random_state 42, test accuracy 0.996124
  (99.61%).
- This is higher than the V2 recorded test accuracy on the supplied metadata. It does not
  alone prove better generalization; the test split and evaluation protocol matter.

### V4 — XGBoost classifier experiment
- `artifacts/modelA_xgb_v4.json/.pkl`
- Metadata reports XGBoost using feature version 3 (95 features), 29 classes,
  1,290 samples (1,032 train / 258 test), random_state 42, test accuracy 0.972868
  (97.29%).
- The recorded score is below the V3 Random Forest score on the supplied metadata.
  This supports saying this XGBoost run scored lower on that test set; it does not establish
  a universal weakness of XGBoost.

### V5 — webcam/inference iteration
- `code/webcam_v5.py`: historical webcam implementation.
- `webcam.py` remains in the active parent directory as the generic webcam script.
- The supplied filenames and source establish that V5 had a dedicated webcam script.
  They do not provide a standalone V5 evaluation report or measured webcam accuracy.
- Do not claim V5 failed without runtime logs or a documented failure.

### V7 — canonicalized representation experiment
- `code/analyze_v7.py`, `code/prepare_data_v7.py`, `code/train_v7.py`,
  `code/validate_v7.py`
- `data/modelA_v7_canonical.npz` and metadata.
- `artifacts/modelA_rf_v7.json/.pkl`
- Artifact metadata reports RandomForest, feature version 3, 95 features, 29 classes,
  53,271 train / 13,318 test samples, 300 estimators, random_state 42,
  class_weight balanced, max_features sqrt, and accuracy 0.976573 (97.66%).
- Metadata explicitly describes V7 as hand-relative rotation canonicalization and says it
  used the same samples/labels, stratified split configuration and RF configuration as V6.
- Compared with V6's recorded 98.83%, V7's recorded 97.66% is lower by about 1.17
  percentage points. This is an observed result, not proof of the cause. The V7 report/code
  should be consulted before attributing the difference to canonicalization alone.

## Shared code relationships
- `prepare_data.py` is the earlier general versioned extraction script; archived once rather
  than copied into each version.
- `webcam_v5.py` is retained once as the V5 webcam snapshot. The generic `webcam.py` stays
  in the parent because it supports feature-version-aware inference and may be part of V6.
- `train.py` stays in the parent because it supports feature versions including V6; do not
  duplicate or move it merely to represent older training runs.
- `features.py`, `landmarker.py`, and `model.py` stay in the parent as shared/current
  implementation dependencies.

## What is not included / evidence gaps
- V6 is intentionally not archived: it is the active/current baseline in this snapshot.
- V1/V2/V3 JSON dataset sidecars in the supplied ZIP were only 3 bytes (effectively empty);
  version facts above therefore rely on model artifact metadata and script contents.
- No separate V5 model checkpoint or webcam performance log was found in the supplied
  snapshot.
- No V1 trained model artifact was found.
- No separate V2/V3/V4/V7 training log was found; artifact JSON metadata is the available
  recorded evidence.
- Large original image datasets are not copied into this archive. V6 checkpoint shards are
  retained in the parent.
- No claim is made that every historic experiment's failure cause is known. Use "not
  established by available evidence" rather than inventing causes.

## Installation (Windows)
1. Back up the existing `backend/modelA` directory or commit the current state to Git.
2. Extract this package into `G:\\EasternNova\\HandLex\\backend\\modelA\\`.
3. Merge the included `archive/` folder with the existing folder (do not overwrite active
   V6 files).
4. After confirming the archive contents, move only the listed historical source/data/
   artifact files from their original locations into the corresponding `archive/` paths.
   The package already contains copies, so you can verify them before removing originals.
5. Run the active V6 smoke test / existing project test commands from the project root.
6. Only after successful verification, optionally remove the original historical duplicates.

## Rollback
The original ZIP is the rollback source. Keep it unchanged until the active V6 pipeline has
been tested successfully. This archive package itself does not modify the original project.
