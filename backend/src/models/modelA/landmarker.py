from __future__ import annotations

from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def create_landmarker(model_path: str | Path):
    model_path = Path(model_path)
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise FileNotFoundError(f"Invalid Hand Landmarker asset: {model_path}")

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_file(detector, image_path: str | Path):
    image = mp.Image.create_from_file(str(image_path))
    return detector.detect(image)


def get_first_hand(result):
    if not result.hand_landmarks:
        return None, None
    landmarks = result.hand_landmarks[0]
    handedness = None
    if getattr(result, "handedness", None):
        try:
            handedness = result.handedness[0][0].category_name
        except (IndexError, AttributeError):
            handedness = None
    return landmarks, handedness
