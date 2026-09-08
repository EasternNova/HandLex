from __future__ import annotations

from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def create_landmarker(model_path: str | Path):
    """
    Create a MediaPipe Hand Landmarker.

    Parameters
    ----------
    model_path:
        Path to the pretrained hand_landmarker.task file.

    Returns
    -------
    MediaPipe HandLandmarker
        Configured hand landmark detector.
    """

    model_path = Path(model_path)

    if not model_path.exists() or model_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Invalid Hand Landmarker asset:\n{model_path}"
        )

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.HandLandmarker.create_from_options(options)


def detect_file(detector, image_path: str | Path):
    """
    Run hand-landmark detection on one image.

    Parameters
    ----------
    detector:
        MediaPipe Hand Landmarker instance.

    image_path:
        Path to the input image.

    Returns
    -------
    MediaPipe detection result.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = mp.Image.create_from_file(
        str(image_path)
    )

    return detector.detect(image)


def get_first_hand(result):
    """
    Extract the first detected hand and its handedness.

    Returns
    -------
    tuple
        (landmarks, handedness)

        If no hand is detected:
            (None, None)
    """

    if not result.hand_landmarks:
        return None, None

    landmarks = result.hand_landmarks[0]

    handedness = None

    if getattr(result, "handedness", None):
        try:
            handedness = (
                result.handedness[0][0].category_name
            )
        except (IndexError, AttributeError):
            handedness = None

    return landmarks, handedness