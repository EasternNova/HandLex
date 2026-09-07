# HandLex ModelA — six-version landmark pipeline

ModelA is the alphabet-level component of HandLex.

Pipeline:
ASL image/webcam -> MediaPipe Hand Landmarker -> landmarks/features -> classifier -> stable character -> text.

## Versions
1. Baseline: raw MediaPipe x/y/z landmarks (63 features).
2. Geometry normalization: wrist-relative + scale-normalized landmarks (63).
3. Robust features: V2 + finger joint angles + wrist distances + hand-present + handedness (95).
4. Classifier comparison: Random Forest vs XGBoost using the same split.
5. Real-time stability: temporal majority smoothing + confidence output.
6. Text layer: A-Z + space + delete + nothing -> character buffer.

J and Z are dynamic signs; a static-frame classifier is not a complete solution for them.
Version 5 improves stability but does not magically add true motion recognition.

## From backend/src/models/modelA
Run with:
python prepare_data.py --version 3 --limit-per-class 50 --output data/modelA_v3_50.npz
python train.py --data data/modelA_v3_50.npz --classifier rf --output checkpoints/modelA_v3_rf.pkl

Install XGBoost for V4:
pip install xgboost

Then:
python train.py --data data/modelA_v3_50.npz --classifier xgb --output checkpoints/modelA_v3_xgb.pkl

For the full dataset remove --limit-per-class.
