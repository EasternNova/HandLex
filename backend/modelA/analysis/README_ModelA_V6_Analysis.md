# Model A V6 — Analysis and Audit Notes

## Purpose

This folder contains analysis scripts used to inspect the existing Model A V6 dataset, classifier behaviour, feature contributions, orientation sensitivity, and selected classification errors.

**These analyses did not train a new classifier. V6 was not modified.** The canonicalization experiments measured properties of transformed landmark geometry and its compatibility with the existing V6 classifier; they did not replace the production feature pipeline.

## Protected V6 assets

Do not overwrite, retrain, or modify these assets as part of the analysis work:

- `backend/modelA/artifacts/modelA_rf_v6.pkl`
- `backend/modelA/data/modelA_v6_full.npz`
- `backend/modelA/data/modelA_v6_full.json` — currently damaged; see the metadata issue below
- `backend/modelA/data/v6_test.npz`
- `backend/modelA/data/v6_test.json`
- `backend/modelA/checkpoints/v6_checkpoints/`

The archive of older experiments is also outside the scope of this work and should remain unchanged.

## Analysis folder

The analysis scripts have been organized under:

```text
backend/modelA/analysis/
```

The scripts reviewed and run include:

- `analyze_v6.py`
- `analyze_detection_failures.py`
- `analyze_nm_confidence.py`
- `analyze_nm_errors.py`
- `audit_v6_nothing.py`
- `canonicalization_audit_v6.py`
- `canonicalization_classification_v6.py`
- `canonicalized_separability_v6.py`
- `feature_importance_v6.py`
- `orientation_audit_v6.py`
- `permutation_v6.py`
- `rotation_sensitivity_v6.py`
- `make_nm_sheet.py`

The N/M contact-sheet output was created at:

```text
backend/modelA/analysis/reports/nm_error_images.jpg
```

It contains 29 images corresponding to the observed N/M classification errors.

## V6 baseline

The existing V6 dataset and classifier audit reported:

- Samples: 66,589
- Features: 95
- Classes: 29
- Random Forest: 300 trees
- Test samples: 13,318
- Overall test accuracy: 98.8287%
- Total test errors: 156

The `nothing` class has 3,000 samples; 2,999 have all-zero feature vectors and one has a non-zero vector. On the test set, the `nothing` class had 600 samples and 99.8333% accuracy (one `nothing → C` error). Excluding `nothing`, accuracy was 98.7813% across 12,718 samples, with 155 errors.

The 95-feature representation consists of 63 normalized XYZ landmark values, 10 joint angles, 20 wrist-distance values, a hand-present flag, and a handedness value.

## N/M error analysis

The N/M analysis found 29 confusions:

- N predicted as M: 20
- M predicted as N: 9

The contact sheet for these errors was generated successfully:

`backend/modelA/analysis/reports/nm_error_images.jpg`

## Feature contribution analysis

The feature-importance analysis reported the following grouped contributions:

| Feature group | Importance |
|---|---:|
| Normalized landmarks | 55.589% |
| Joint angles | 13.574% |
| Distances | 30.723% |
| Hand-present flag | 0.083% |
| Handedness | 0.030% |

The grouped permutation test on the held-out test set reported:

| Feature group | Accuracy after permutation | Accuracy drop |
|---|---:|---:|
| Normalized landmarks | 53.814% | 45.014 percentage points |
| Joint angles | 95.645% | 3.184 percentage points |
| Distances | 91.132% | 7.696 percentage points |
| Hand-present flag | 98.829% | 0.000 percentage points |
| Handedness | 98.829% | 0.000 percentage points |

These results indicate that normalized landmarks contributed most strongly in these tests, followed by wrist distances and joint angles.

## Orientation and rotation sensitivity

The orientation audit measured mean wrist-to-middle-MCP axis angles of:

- X: 70.18°
- Y: 28.59°
- Z: 80.99°

The rotation-sensitivity test reported hand-present baseline accuracy of 99.7547%. Accuracy decreased as rotations increased:

| Rotation axis | 15° | 30° | 45° | 60° |
|---|---:|---:|---:|---:|
| X | 96.554% | 77.040% | 60.514% | 47.660% |
| Y | 99.329% | 96.787% | 91.626% | 86.265% |
| Z | 95.098% | 59.630% | 27.452% | 20.569% |

This indicates substantial sensitivity to X- and Z-axis rotations under the tested transformations.

## Canonicalization experiments

The canonicalization audit successfully transformed 63,590 hand-present samples. It reported mean pairwise-distance difference of approximately 0.00000002 and maximum difference of approximately 0.00000036, consistent with preservation of pairwise geometry under the tested rigid rotations.

However, feeding canonicalized features into the existing V6 classifier produced:

- Original V6 hand-present accuracy: 99.7547%
- Canonicalized-input accuracy: 15.5575%
- Prediction change rate: 84.4394%

**Conclusion:** the experimental canonicalized representation is not compatible with the existing V6 classifier. Do not insert this canonicalization into the live webcam or production inference pipeline without a separately designed and validated model/pipeline.

The canonicalized separability experiment reported a silhouette score of 0.199504 on 5,000 sampled points. Its reported 98.45% nearest-neighbour same-class rate requires re-evaluation: the script fitted the neighbour search on the same data it queried, so a sample could select itself as its nearest neighbour. Treat that rate as potentially inflated until the evaluation excludes self-neighbours or uses a separate train/query split.

## Detection-failure audit — intentionally skipped

The optional `analyze_detection_failures.py` audit was stopped because its metadata input, `backend/modelA/data/modelA_v6_full.json`, contains only a UTF-8 BOM (3 bytes) and no JSON data.

The V6 NPZ dataset remains readable and contains `X`, `y`, `paths`, `classes`, and `version` for 66,589 saved samples. Those successful samples do not establish the number of source images that failed detection.

**Decision:** this audit is not required for the current Model A completion work. It was intentionally skipped; no metadata reconstruction, dataset regeneration, or classifier retraining will be performed for it. Do not treat detection-failure counts or rates as verified results.

The `analyze_detection_failures.py` script is therefore excluded from the completed analysis workflow. All other completed analyses remain separate from this skipped audit.

## Running the scripts

Run commands from the HandLex project root, for example:

```powershell
py backend/modelA/analysis/analyze_v6.py
py backend/modelA/analysis/audit_v6_nothing.py
py backend/modelA/analysis/feature_importance_v6.py
py backend/modelA/analysis/permutation_v6.py
py backend/modelA/analysis/rotation_sensitivity_v6.py
py backend/modelA/analysis/make_nm_sheet.py
```

The contact-sheet script is located directly in `analysis`, while its generated image is saved in `analysis/reports`.

## Environment note

The saved scikit-learn model was created with version 1.9.0, while the current environment has version 1.9.1. A version-mismatch warning was observed, but the model loaded and the analysis scripts ran. Do not change package versions solely to suppress this warning without a separate compatibility decision.

## Current status

- [x] Analysis scripts organized under `backend/modelA/analysis/`
- [x] V6 baseline analysis completed
- [x] `nothing` class audit completed
- [x] N/M error analysis completed
- [x] N/M contact sheet generated
- [x] Feature importance and permutation analyses completed
- [x] Orientation and rotation sensitivity analyses completed
- [x] Canonicalization geometry and classifier-compatibility experiments completed
- [x] Optional detection-failure audit intentionally skipped; metadata repair not required
- [ ] Correct the nearest-neighbour separability evaluation to exclude self-matches

**Final safeguard:** no classifier was trained and the V6 classifier was not modified during this analysis work.
