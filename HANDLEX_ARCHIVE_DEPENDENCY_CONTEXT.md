# HandLex — dependency and archive inspection context

## What was inspected

This note is based on the uploaded `frontend.zip` and `backend.zip`.

## Frontend archive

The frontend is a React application built with Vite.

### Existing direct dependencies (`frontend/package.json`)

Runtime:
- `react`
- `react-dom`
- `react-router-dom`

Development/build:
- `vite`
- `@vitejs/plugin-react`
- `eslint`
- `@eslint/js`
- `eslint-plugin-react-hooks`
- `eslint-plugin-react-refresh`
- `globals`
- `@types/react`
- `@types/react-dom`

The archive also includes `frontend/package-lock.json`. Keep it committed and use
`npm ci` for reproducible installs. The frontend does not need a Python
`requirements.txt`.

### Existing pages

The source includes Home, Translate, Learn, Alphabet, Words, Phrases, Progress,
Practice, Dictionary, and About pages. The Translate page currently accesses the
camera in the browser with `navigator.mediaDevices.getUserMedia`.

The inspected frontend package manifest does not list an HTTP client such as Axios.
When backend integration is added, the browser can use native `fetch()` or a new
client library if needed.

## Backend archive

The uploaded backend currently contains Python scripts for Model A and archived
Model B checkpoint files.

### Model A — direct third-party dependencies found in Python imports

| Package | Purpose |
|---|---|
| `mediapipe` | Hand Landmarker and landmark detection |
| `opencv-python` | Webcam/video frame capture and image processing |
| `Pillow` | Image loading / image-sheet utilities |
| `numpy` | Feature arrays and `.npz` datasets |
| `scikit-learn` | Random Forest, evaluation, data splitting |
| `joblib` | Saving/loading trained model artifacts |
| `xgboost` | XGBoost classifier used by the training script |

The generated `backend/requirements.txt` contains this current direct dependency set.
It intentionally avoids speculative packages and unverified version pins.

### Model B — important current limitation

The archive contains archived WLASL `.pt` checkpoint files and matching `.ini`
configuration files under `backend/modelB/archived/` (ASL100, ASL300, ASL1000,
ASL2000). However, the uploaded archive does not contain Model B Python source code
or imports that establish its runtime requirements. PyTorch is therefore not added
to the current requirements file yet. Add the exact PyTorch/torchvision and other
dependencies after identifying the checkpoint architecture, preprocessing contract,
and inference implementation.

### Backend API — not present yet

No FastAPI, Flask, or other HTTP API server implementation was found in the uploaded
backend Python files. The current frontend and backend are therefore not yet connected
through an implemented API. When we build that layer, choose an API framework
(e.g. FastAPI) and add its packages to the backend requirements.

## Suggested project layout

```text
HandLex/
├── frontend/
│   ├── package.json
│   ├── package-lock.json
│   └── src/
│       ├── App.jsx
│       └── pages/
└── backend/
    ├── requirements.txt
    ├── modelA/
    │   ├── features.py
    │   ├── landmarker.py
    │   ├── model.py
    │   ├── prepare_data.py
    │   ├── train.py
    │   └── webcam.py
    └── modelB/
        └── archived/
            ├── asl100/
            ├── asl300/
            ├── asl1000/
            └── asl2000/
```

## Install commands

Backend (run inside `backend/`):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Frontend (run inside `frontend/`):

```powershell
npm ci
npm run dev
```

Build frontend:

```powershell
npm run build
```

## Deployment boundary

- Vercel is suitable for deploying the React/Vite frontend.
- The Python model runtime should be deployed separately on a Python-capable host.
- The browser camera capture in `Translate.jsx` is not itself model inference; it still
  needs a secure, designed connection to the backend (for example, frame/image upload
  or a streaming transport) before predictions can appear in the UI.
- Configure the deployed frontend's backend API URL using an environment variable.
  Configure backend CORS to allow the deployed frontend origin.

## Important cleanup observations

- The uploaded frontend archive includes `node_modules/`; do not upload that folder to
  GitHub or Vercel as source. Keep `package-lock.json` and install with `npm ci`.
- The backend archive includes large trained artifacts/checkpoints and data files.
  Review Git LFS or external model storage before pushing large binary files to GitHub.
- The archive contains Python `__pycache__` files; these are generated files and should
  not be committed.
- Do not treat the previous generic dependency list as a verified project manifest.
  This version is based on imports and manifests actually present in these archives.
