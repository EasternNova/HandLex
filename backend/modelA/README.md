# HandLex Model A

## Purpose
Model A is the static hand-sign recognition model used within the HandLex translator.

## Core Components

| File | Responsibility |
|---|---|
| `features.py` | Feature extraction |
| `landmarker.py` | Hand landmark detection |
| `model.py` | Model-related definitions |
| `webcam.py` | Live inference and webcam testing |
| `test_hand_landmarker.py` | Hand detector testing |

## Training & Data

| File | Responsibility |
|---|---|
| `prepare_data_v6.py` | Dataset preparation |
| `train.py` | Model training |

## Analysis
All model evaluation, diagnostics, experimental analysis, and reports are stored in `analysis/`.

## Model Artifacts
The trained V6 model is stored in `artifacts/`.

## Dataset
The dataset and testing data are stored in `data/`.

## Checkpoints
Training checkpoints are stored in `checkpoints/`.

## Archive
Historical code, models, datasets and experiments are preserved in `archive/`.

## Current Production Candidate
Model A V6 — Random Forest classifier, 95 features, 29 classes.

## Important
- Do not modify the V6 production artifact during organization.
- Do not modify the original dataset or archived files.
- Experimental canonicalization is not used in production inference.
- Model A will eventually integrate into the unified HandLex Translator alongside other models.

# Model A — HandLex

## Current Development Status

### Completed: Model A V6 Inference Engine

The original Model A V6 classifier has been successfully connected to a reusable inference engine.

**Verified accomplishments:**

- Loaded the existing trained Random Forest classifier.
- Loaded the MediaPipe HandLandmarker asset.
- Initialized the inference engine successfully.
- Verified V3 feature configuration (95 features).
- Successfully detected a hand in a test image.
- Extracted the 95-feature input vector.
- Successfully generated a prediction using `predict_proba()`.
- Generated Top-3 predictions.
- Returned structured inference results.
- Verified the engine cleanup operation.

### Model A V6 Configuration

| Property | Value |
|---|---|
| Model | Model A V6 |
| Classifier | Random Forest |
| Feature version | V3 |
| Feature count | 95 |
| Classes | 29 |
| Model artifact | `artifacts/modelA_rf_v6.pkl` |
| Hand detection | MediaPipe HandLandmarker |
| HandLandmarker asset | `backend/models/pretrained/hand_landmarker.task` |

### First Successful Inference Test

**Test image:**

`backend/modelA/test_images/test_hand.jpg`

**Result:**

- Hand detected: Yes
- Handedness: Right
- Prediction: `10`
- Confidence: `1.0`
- Top-3 predictions: `10`, `27`, `26`

The first successful image inference confirms that the classifier, landmark detection, feature extraction and prediction pipeline are operational.

This is an initial functional test, not a complete accuracy or performance validation.

---

## C. Prepare Model A for the Backend

- Create a Python API, likely using FastAPI.
- Create prediction endpoints.
- Define how camera frames and data are received.
- Define structured prediction responses.
- Handle errors and model loading safely.

## D. Connect the Frontend

- Build the Model A translator page.
- Connect the frontend camera.
- Send frames or landmarks to the backend.
- Display live predictions and confidence.
- Display translated text.
- Implement clear, space and delete.
- Implement camera start/stop.

## E. Final Testing

- Test isolated predictions.
- Test webcam performance.
- Test incorrect predictions.
- Test no-hand scenarios.
- Test frontend/backend communication.
- Test the complete translator workflow.

---

## 5. Our Working Order From Now On

We will follow this order rather than randomly editing files.

### STEP 1 — Cleanup & Audit
Remove duplicates, establish proper naming and verify existing code.

**Status: Completed**

### STEP 2 — Organize Model A
Create the proper folder structure and relocate existing files.

**Status: Completed**

### STEP 3 — Build Inference Engine
Separate model loading, features, prediction and smoothing.

**Status: Initial inference engine implemented and initialized.**

### STEP 4 — Test Inference
Ensure the cleaned code still correctly uses V6.

**Status: Initial image inference passed. Further validation is pending.**

### STEP 5 — Build Backend API
Expose Model A functionality to the website.

**Status: Pending**

### STEP 6 — Connect Frontend
Connect webcam, prediction, confidence and text output.

**Status: Pending**

### STEP 7 — Final Integration
Test HandLex Model A from the website, end to end.

**Status: Pending**

---

## Immediate Next Task

### STEP 3 — Inference Validation and Label Mapping

Before building the backend API, validate the inference engine and verify the mapping between numeric class IDs and actual sign labels.

Planned work:

1. Verify the class-label mapping.
2. Test multiple sign images.
3. Check prediction consistency.
4. Validate feature extraction.
5. Identify incorrect predictions.
6. Confirm no-hand behavior.
7. Prepare the engine for continuous frame inference and temporal smoothing.

**Next milestone:** A validated and reliable Model A inference engine ready to be exposed through the backend API.

---

## Development Principle

Preserve the original Model A V6 training artifacts.

All inference, validation and backend work must use the existing trained model without unnecessarily retraining, overwriting or modifying the original training data, checkpoints or classifier artifacts.