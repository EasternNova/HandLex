import cv2
import torch
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from backend.src.models.modelB.model import ModelB


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "backend/checkpoints/modelB/best_model.pth"
HOLISTIC_PATH = "dataset/WLASL/models/holistic_landmarker.task"
CLASS_LIST = "dataset/WLASL/wlasl_class_list.txt"


# ============================================================
# LOAD WLASL CLASSES
# ============================================================

def load_classes():
    classes = {}

    with open(CLASS_LIST, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:
                idx, name = parts
                classes[int(idx)] = name

    return classes


# ============================================================
# MEDIAPIPE HOLISTIC LANDMARKER
# ============================================================

def create_landmarker():

    base_options = python.BaseOptions(
        model_asset_path=HOLISTIC_PATH
    )

    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    return vision.HolisticLandmarker.create_from_options(
        options
    )


# ============================================================
# EXTRACT 75 KEYPOINTS
# ============================================================

def extract_keypoints(result):

    keypoints = []

    # --------------------------------------------------------
    # Pose: 33
    # --------------------------------------------------------

    if result.pose_landmarks:
        for lm in result.pose_landmarks:
            keypoints.append([lm.x, lm.y])

    if len(keypoints) < 33:
        keypoints.extend(
            [[0.0, 0.0]] * (33 - len(keypoints))
        )

    # --------------------------------------------------------
    # Left hand: 21
    # --------------------------------------------------------

    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks:
            keypoints.append([lm.x, lm.y])

    if len(keypoints) < 54:
        keypoints.extend(
            [[0.0, 0.0]] * (54 - len(keypoints))
        )

    # --------------------------------------------------------
    # Right hand: 21
    # --------------------------------------------------------

    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks:
            keypoints.append([lm.x, lm.y])

    if len(keypoints) < 75:
        keypoints.extend(
            [[0.0, 0.0]] * (75 - len(keypoints))
        )

    return np.array(
        keypoints[:75],
        dtype=np.float32
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_keypoints(data, selected):

    tensor = torch.from_numpy(
        data
    ).float()

    for i in range(tensor.shape[0]):

        part = tensor[
            i,
            selected,
            :
        ]

        min_x = part[:, 0].min()
        max_x = part[:, 0].max()

        min_y = part[:, 1].min()
        max_y = part[:, 1].max()

        w = max_x - min_x
        h = max_y - min_y

        if w > h:

            dx = 0.05 * w
            dy = dx + ((w - h) / 2)

        else:

            dy = 0.05 * h
            dx = dy + ((h - w) / 2)

        sx = max(
            0,
            min(min_x - dx, 1)
        )

        sy = max(
            0,
            min(min_y - dy, 1)
        )

        ex = max(
            0,
            min(max_x + dx, 1)
        )

        ey = max(
            0,
            min(max_y + dy, 1)
        )

        if ex - sx != 0:

            part[:, 0] = (
                part[:, 0] - sx
            ) / (ex - sx)

        if ey - sy != 0:

            part[:, 1] = (
                part[:, 1] - sy
            ) / (ey - sy)

        tensor[
            i,
            selected,
            :
        ] = part

    return tensor


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("HANDLEX - MODEL B WEBCAM INFERENCE")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    config = checkpoint["config"]

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Validation accuracy: "
        f"{checkpoint.get('val_acc', 0) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ModelB(config).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    classes = load_classes()

    print(
        f"Classes loaded: {len(classes)}"
    )

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    landmarker = create_landmarker()

    print("MediaPipe loaded successfully.")

    # --------------------------------------------------------
    # Webcam
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        landmarker.close()
        return

    print()
    print("Webcam started.")
    print("Perform a sign in front of the camera.")
    print("Press Q to quit.")
    print()

    sequence = []
    frame_count = 0

    # ModelB uses 64-frame sequences
    sequence_length = 64

    # --------------------------------------------------------
    # Webcam loop
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Failed to read webcam frame."
            )

            break

        frame_count += 1

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = frame_count * 33

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        # ----------------------------------------------------
        # Keypoints
        # ----------------------------------------------------

        kp = extract_keypoints(result)

        sequence.append(kp)

        if len(sequence) > sequence_length:

            sequence.pop(0)

        prediction_text = (
            "Collecting frames..."
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if len(sequence) >= sequence_length:

            data = np.stack(
                sequence
            )

            data = np.clip(
                data,
                0,
                1
            )

            tensor = normalize_keypoints(
                data,
                config["joint_idx"]
            )

            tensor = tensor.unsqueeze(0)

            attention_mask = torch.ones(
                (1, tensor.shape[1]),
                dtype=torch.long
            )

            tensor = tensor.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():

                _, logits = model(
                    keypoints=tensor,
                    attention_mask=attention_mask,
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )

                confidence, prediction = (
                    probabilities.max(dim=1)
                )

                class_id = prediction.item()

                word = classes.get(
                    class_id,
                    "Unknown"
                )

                prediction_text = (
                    f"{word} "
                    f"({confidence.item() * 100:.1f}%)"
                )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "HandLex - MODEL B",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            prediction_text,
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Frames: {len(sequence)}/{sequence_length}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "HandLex",
            frame
        )

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()

    landmarker.close()

    cv2.destroyAllWindows()

    print()
    print("Webcam stopped.")


if __name__ == "__main__":
    main()