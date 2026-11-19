# HandLex — Backend and Frontend Dependency Context

## Architecture

- **Frontend:** React + Vite. It is the user interface and should be deployed from the
  `frontend/` directory to Vercel (`npm install`, `npm run build`, output `dist`).
- **Backend:** Python + FastAPI. It exposes REST/JSON endpoints to the frontend and
  runs the recognition pipeline.
- **Model A:** ASL alphabet recognition using MediaPipe hand landmarks and a
  scikit-learn Random Forest classifier. The current feature-extraction work uses
  63-dimensional feature vectors in the V1 pipeline.
- **Model B:** WLASL word recognition. I3D is the primary planned/runtime architecture;
  PyTorch is included. Its original preprocessing and checkpoint compatibility must
  be preserved and verified.
- **Model C:** Continuous signing / sentence recognition using How2Sign is planned.
  Its final architecture and exact dependencies have not yet been established.

## Which dependency file belongs where?

| Area | Dependency manager | File |
|---|---|---|
| Python API and ML models | pip | `backend/requirements.txt` |
| React + Vite interface | npm | `frontend/package.json` |

A Python `requirements.txt` does **not** install React, Vite, or browser-side JavaScript
packages. Those belong in `frontend/package.json`.

## Install Python dependencies

Run from the directory containing `requirements.txt`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For PyTorch on a CPU-only Windows machine, use the CPU wheel selector/instructions from
the official PyTorch installation page if pip selects an unsuitable build:
https://pytorch.org/get-started/locally/

## Frontend install and run

```powershell
cd frontend
npm install
npm run dev
```

## Important implementation notes

1. Keep the frontend and backend dependency manifests separate.
2. The frontend should call the FastAPI backend through an environment-configured API URL;
   do not hard-code a local machine URL in production.
3. Configure FastAPI CORS for the deployed Vercel frontend origin.
4. Vercel is currently the frontend deployment target. The Python ML backend is a separate
   deployment and is not made deployable merely by uploading this requirements file to Vercel.
5. Model A and Model B dependencies are included because those model tracks are part of the
   current HandLex plan. Model C-specific packages should be added only after its architecture
   and implementation are selected.
6. The version ranges are compatibility bounds, not a verified lockfile. Once the project
   installs and passes tests in its chosen Python environment, freeze exact versions into a
   lock/constraints file for reproducible deployment.
7. Python 3.11 or 3.12 is recommended for the ML environment. Check wheel availability before
   using Python 3.15, especially for MediaPipe and PyTorch.
