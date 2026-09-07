from pathlib import Path

import cv2
import mediapipe as mp


# --------------------------------------------------
# PATH
# --------------------------------------------------

DATA_DIR = (
    Path(__file__).resolve().parents[4]
    / "dataset"
    / "ASL"
    / "asl_alphabet_train"
    / "asl_alphabet_train"
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MAX_IMAGES = 50


# --------------------------------------------------
# MEDIAPIPE
# --------------------------------------------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)


# --------------------------------------------------
# DATASET CHECK
# --------------------------------------------------

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_DIR}"
    )

class_dirs = sorted(
    p for p in DATA_DIR.iterdir()
    if p.is_dir()
)

print("Dataset:", DATA_DIR)
print("Classes:", len(class_dirs))
print("First classes:", [p.name for p in class_dirs[:5]])


# --------------------------------------------------
# TEST LANDMARK EXTRACTION
# --------------------------------------------------

images_checked = 0
hands_detected = 0
hands_missing = 0

feature_lengths = set()


for class_dir in class_dirs:

    image_paths = list(class_dir.glob("*.jpg"))

    for image_path in image_paths:

        if images_checked >= MAX_IMAGES:
            break

        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not read:", image_path)
            continue

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        images_checked += 1

        if not results.multi_hand_landmarks:
            hands_missing += 1
            continue

        hands_detected += 1

        landmarks = results.multi_hand_landmarks[0]

        # 21 landmarks × (x, y)
        features = []

        for landmark in landmarks.landmark:
            features.append(landmark.x)
            features.append(landmark.y)

        feature_lengths.add(len(features))


    if images_checked >= MAX_IMAGES:
        break


hands.close()


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print()
print("========== MEDIAPIPE TEST ==========")
print("Images checked:", images_checked)
print("Hands detected:", hands_detected)
print("No hand detected:", hands_missing)
print("Feature lengths:", feature_lengths)

if images_checked == 0:
    raise RuntimeError("No images were processed.")

if hands_detected == 0:
    raise RuntimeError(
        "MediaPipe detected no hands in the test images."
    )

if feature_lengths != {42}:
    raise RuntimeError(
        f"Unexpected feature length: {feature_lengths}"
    )

print()
print("LANDMARK EXTRACTION TEST: PASSED")