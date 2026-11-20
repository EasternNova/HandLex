
"""
HandLex - Model A V6 Inference Engine

Provides reusable single-frame and image sign prediction.
Does not modify the trained model or dataset.
"""

from pathlib import Path

import joblib
import numpy as np
import cv2
import mediapipe as mp

from .features import make_feature_vector
from .landmarker import create_landmarker



# PATHS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "modelA"
    / "artifacts"
    / "modelA_rf_v6.pkl"
)

LANDMARKER_PATH = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "pretrained"
    / "hand_landmarker.task"
)



# MODEL A INFERENCE ENGINE


class ModelAInference:

    def __init__(self):

        # Validate required assets
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        if not LANDMARKER_PATH.exists():
            raise FileNotFoundError(
                f"HandLandmarker not found: {LANDMARKER_PATH}"
            )

        # Load the trained classifier package
        self.package = joblib.load(MODEL_PATH)

        self.model = self.package["model"]
        self.classes = self.package["classes"]
        self.feature_version = self.package["feature_version"]
        self.feature_count = self.package["feature_count"]

        # Verify V6 feature configuration
        if self.feature_version != 3:
            raise ValueError(
                f"Expected feature version 3, "
                f"got {self.feature_version}"
            )

        if self.feature_count != 95:
            raise ValueError(
                f"Expected 95 features, "
                f"got {self.feature_count}"
            )

        # Initialize MediaPipe HandLandmarker
        self.detector = create_landmarker(
            str(LANDMARKER_PATH)
        )

        print("Model A V6 loaded successfully.")
        print(f"Feature version: {self.feature_version}")
        print(f"Feature count: {self.feature_count}")
        print(f"Classes: {len(self.classes)}")


    
    # LANDMARK DETECTION
    

    def detect_hand(self, frame):

        # Convert OpenCV BGR frame to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.detector.detect(mp_image)

        # No hand detected
        if not result.hand_landmarks:
            return None, None

        # Use the first detected hand
        landmarks = result.hand_landmarks[0]

        handedness = None

        if result.handedness:
            handedness = (
                result.handedness[0][0].category_name
            )

        return landmarks, handedness


    
    # FEATURE EXTRACTION
    

    def extract_features(self, landmarks, handedness):

        features = make_feature_vector(
            landmarks=landmarks,
            handedness=handedness,
            version=3
        )

        features = np.asarray(
            features,
            dtype=np.float32
        ).reshape(1, -1)

        # Verify feature dimensions
        if features.shape[1] != 95:
            raise ValueError(
                f"Unexpected feature shape: {features.shape}"
            )

        return features


    
    # PREDICTION
    

    def predict_frame(self, frame):

        landmarks, handedness = self.detect_hand(frame)

        # No hand detected
        if landmarks is None:

            return {
                "prediction": "nothing",
                "confidence": 1.0,
                "top3": [],
                "hand_detected": False,
                "handedness": None
            }

        # Extract V6 features
        features = self.extract_features(
            landmarks,
            handedness
        )

        # Predict class probabilities
        probabilities = self.model.predict_proba(
            features
        )[0]

        # Find predicted class index
        predicted_index = int(
            np.argmax(probabilities)
        )

        # Convert numeric class ID to actual label
        class_id = int(
            self.model.classes_[predicted_index]
        )

        prediction = str(
            self.classes[class_id]
        )

        confidence = float(
            probabilities[predicted_index]
        )

        # Generate Top-3 predictions
        top_indices = np.argsort(
            probabilities
        )[::-1][:3]

        top3 = []

        for i in top_indices:

            class_id = int(
                self.model.classes_[i]
            )

            top3.append({
                "sign": str(self.classes[class_id]),
                "confidence": float(probabilities[i])
            })

        # Structured prediction response
        return {
            "prediction": prediction,
            "confidence": confidence,
            "top3": top3,
            "hand_detected": True,
            "handedness": handedness
        }


    
    # IMAGE INFERENCE
    

    def predict_image(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        return self.predict_frame(frame)


    
    # CLEANUP
    

    def close(self):

        if hasattr(self.detector, "close"):
            self.detector.close()
