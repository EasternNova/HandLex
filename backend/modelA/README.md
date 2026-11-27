# Model A — HandLex

## Overview

Model A is HandLex's currently integrated hand-sign classifier. Its active inference pipeline uses MediaPipe to detect hand landmarks, extracts a 95-feature representation, and passes the features to a saved Random Forest model.

## Current model

| Property | Value |
|---|---|
| Classifier | Random Forest |
| Artifact | `artifacts/modelA_rf_v6.pkl` |
| Feature version | 3 |
| Feature count | 95 |
| Classes | 29 |
| Recorded test accuracy | 98.8287% |
| Test samples | 13,318 |
| Split | Stratified 80/20; random state 42 |

The benchmark is a recorded evaluation result and should not be interpreted as webcam accuracy.

## Classes

A–Z plus `del`, `nothing`, and `space` (29 total).

## Features

The feature vector includes:

- 63 normalized XYZ landmark values
- 10 joint-angle values
- 20 wrist-distance values
- Hand-present flag
- Handedness information

## Inference flow

```text
Image frame
  ↓
MediaPipe HandLandmarker
  ↓
Landmarks
  ↓
Feature extraction (version 3, 95 values)
  ↓
Saved Random Forest
  ↓
Prediction and confidence
  ↓
Temporal prediction stabilizer
```

## Live integration

Model A is called through the FastAPI service. The API loads the model once in memory and uses the image endpoint for uploaded camera frames.

Relevant files include:

- `inference.py` — inference wrapper
- `features.py` — feature extraction
- `landmarker.py` — landmark detection
- `prediction_stabilizer.py` — temporal stabilization
- `artifacts/` — saved model artifacts and metadata
- `data/` — prepared datasets and test files
- `checkpoints/` — preserved processing checkpoints

## Run the API

From the HandLex root:

```powershell
py -3.15 -m uvicorn backend.api.main:app --reload
```

Then in Swagger (`http://127.0.0.1:8000/docs`):

1. Execute `POST /models/model-a/load`.
2. Use `POST /models/model-a/predict-image` to upload an image.

The frontend Translate page sends camera frames to this endpoint after Model A is loaded.

## Testing notes and limitations

Initial webcam tests have produced predictions (including some letters such as B and C), but recognition is inconsistent across signs. Lighting, hand position, camera angle, landmark detection, and differences between training and real-world images may affect results.

Separate external-image tests and benchmark results should not be conflated with live-webcam performance.

## Preservation

Do not retrain, replace, or overwrite the protected V6 model, datasets, checkpoints, or evaluation records without backups and a deliberate evaluation plan.
