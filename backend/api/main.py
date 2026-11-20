import numpy as np
import cv2

from fastapi import UploadFile, File, HTTPException

from pathlib import Path

from fastapi import FastAPI

from backend.api.model_a_service import load_model_a, get_model_a, unload_model_a

app = FastAPI(
    title="HandLex API",
    description="Backend API for the HandLex sign language translation system",
    version="1.0.0",
)

ROOT = Path(__file__).resolve().parents[2]

MODEL_A_ARTIFACT = ROOT / "backend" / "modelA" / "artifacts" / "modelA_rf_v6.pkl"
HAND_LANDMARKER = ROOT / "backend" / "models" / "pretrained" / "hand_landmarker.task"


@app.get("/")
def root():
    return {"message": "HandLex API is running"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": "HandLex",
        "api": "running",
    }


@app.get("/models/model-a/status")
def model_a_status():
    return {
        "model": "Model A",
        "available": MODEL_A_ARTIFACT.exists() and HAND_LANDMARKER.exists(),
        "model_artifact": MODEL_A_ARTIFACT.exists(),
        "hand_landmarker": HAND_LANDMARKER.exists(),
        "loaded": get_model_a() is not None,
    }


@app.post("/models/model-a/load")
def load_model_a_route():
    model = load_model_a()

    return {
        "message": "Model A loaded successfully",
        "loaded": True,
        "feature_version": model.feature_version,
        "feature_count": model.feature_count,
        "classes": len(model.classes),
    }


@app.post("/models/model-a/unload")
def unload_model_a_route():
    unload_model_a()

    return {
        "message": "Model A unloaded",
        "loaded": False,
    }

@app.post("/models/model-a/predict-image")
async def predict_image_route(file: UploadFile = File(...)):

    model = get_model_a()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model A is not loaded. Load it first."
        )

    image_bytes = await file.read()

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    result = model.predict_frame(frame)

    return result