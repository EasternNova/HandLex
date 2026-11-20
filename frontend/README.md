# HandLex Frontend

The HandLex frontend is built with React and Vite. It provides the website interface, learning pages, and the Translate page that captures camera frames and communicates with the backend API.

## Technology

- React
- Vite
- JavaScript
- CSS

## Setup

From this directory:

```powershell
npm install
npm run dev
```

Use the local URL printed by Vite (typically `http://localhost:5173`).

## Translate page integration

The Translate page:

1. Requests browser camera permission.
2. Displays the webcam stream.
3. Captures periodic frames.
4. Sends JPEG frames as multipart form data to the backend.
5. Displays the response prediction, confidence, and status.

The current API request is directed to:

```text
http://127.0.0.1:8000/models/model-a/predict-image
```

The backend must be running and Model A must be loaded first.

## Backend prerequisites

From the HandLex project root, run the API:

```powershell
py -3.15 -m uvicorn backend.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` and execute `POST /models/model-a/load` before starting translation.

## Frontend scripts

Use the scripts defined in `package.json`. Typical Vite scripts include:

```powershell
npm run dev
npm run build
npm run preview
```

## Current limitations

- Live prediction is under testing.
- Recognition consistency varies across signs and camera conditions.
- The interface currently displays individual predictions; full continuous sentence translation is not implemented.
- The frontend development server and backend API are separate processes.

## Notes

This README replaces the generic React/Vite template description with project-specific information. For the overall project, see the root HandLex README.
