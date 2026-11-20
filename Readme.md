# HandLex API

## Overview

The HandLex API is the Python backend that connects the React/Vite website to the model inference code. It is implemented using FastAPI and runs locally through Uvicorn.

## Start the server

From the HandLex project root:

```powershell
py -3.15 -m uvicorn backend.api.main:app --reload
```

- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome response |
| GET | `/health` | Health check |
| GET | `/models/model-a/status` | File availability and loaded state |
| POST | `/models/model-a/load` | Load Model A |
| POST | `/models/model-a/unload` | Unload Model A |
| POST | `/models/model-a/predict-image` | Image prediction |

## Model lifecycle

1. Start the API.
2. Check `/models/model-a/status`.
3. Load the model using `POST /models/model-a/load`.
4. Send image uploads to `POST /models/model-a/predict-image`.
5. Unload the model with `POST /models/model-a/unload` when needed.

The model is held in backend memory. A server restart clears the loaded state, so loading must be repeated after restart.

## Image prediction

The image endpoint expects an uploaded image using multipart form data. It decodes the image using OpenCV and passes the frame to Model A inference.

The current frontend captures JPEG frames and uploads them to the endpoint.

## CORS

The current development CORS configuration allows the frontend development origins:

- `http://localhost:5173`
- `http://127.0.0.1:5173`

This configuration is for local development. Production deployment should use an appropriately restricted origin configuration.

## Current limitations

- Model A is the only model currently connected to the active API workflow.
- Model B checkpoints are archived and not yet integrated as a runtime.
- Model C is a planned workspace; its runtime is not currently documented.
- The API and frontend are currently used in local development.
- Live recognition results require testing and should not be treated as guaranteed.

## Dependencies

See the backend `requirements.txt` for FastAPI, Uvicorn, multipart handling, image processing, and Model A dependencies.
