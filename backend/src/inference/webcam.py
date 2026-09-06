import cv2
import json
import torch
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.models.modelB.model import SignBart


MODEL_PATH = "checkpoints/modelB/epoch_10.pth"
HOLISTIC_PATH = "data/WLASL/models/holistic_landmarker.task"
CLASS_LIST = "data/WLASL/wlasl_class_list.txt"


def load_classes():
    classes = {}

    with open(CLASS_LIST, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:
                idx, name = parts
                classes[int(idx)] = name

    return classes


def create_landmarker():
    base_options = python.BaseOptions(
        model_bsset_path=HOLISTIC_PATH
    )

    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    return vision.HolisticLandmarker.create_from_options(options)


def extract_keypoints(result):
    keypoints = []

    # Pose: 33
    if result.pose_landmarks:
        for lm in result.pose_landmarks:
            keypoints.append([lm.x, lm.y])

    if len(keypoints) < 33:
        keypoints.extend(
            [[0.0, 0.0]] * (33 - len(keypoints))
        )

    # Left hand: 21
    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks:
            keypoints.append([lm.x, lm.y])

    if len(keypoints) < 54:
        keypoints.extend(
            [[0.0, 0.0]] * (54 - len(keypoints))
        )

    # Right hand: 21
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


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    config = checkpoint["config"]

    model = SignBart(config).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Model loaded.")

    # --------------------------------------------------
    # Classes
    # --------------------------------------------------

    classes = load_classes()

    # --------------------------------------------------
    # MediaPipe
    # --------------------------------------------------

    landmarker = create_landmarker()

    # --------------------------------------------------
    # Webcam
    # --------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print()
    print("Webcam started.")
    print("Press Q to quit.")
    print()

    sequence = []
    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read webcam frame.")
            break

        frame_count += 1

        # OpenCV BGR → RGB
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

        kp = extract_keypoints(result)

        sequence.append(kp)

        # Keep most recent 32 frames
        if len(sequence) > 32:
            sequence.pop(0)

        prediction_text = "Collecting..."

        if len(sequence) >= 8:

            data = np.stack(sequence)

            # --------------------------------------------------
            # Match training preprocessing
            # --------------------------------------------------

            data = np.clip(
                data,
                0,
                1
            )

            tensor = torch.from_numpy(
                data
            ).float()

            # Normalize selected joints
            for i in range(tensor.shape[0]):

                selected = config["joint_idx"]

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

            tensor = tensor.unsqueeze(0)

            attention_mask = torch.ones(
                (1, tensor.shape[1])
            )

            tensor = tensor.to(device)
            attention_mask = attention_mask.to(device)

            # --------------------------------------------------
            # Prediction
            # --------------------------------------------------

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

                prediction_text = (
                    f"{classes.get(class_id, 'Unknown')} "
                    f"({confidence.item() * 100:.1f}%)"
                )

        # --------------------------------------------------
        # Display
        # --------------------------------------------------

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

        cv2.imshow(
            "HandLex",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    landmarker.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()