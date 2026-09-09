import os
import json
import pickle
import argparse

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

VIDEO_DIR = "dataset/WLASL/videos"
MODEL_PATH = "dataset/WLASL/models/holistic_landmarker.task"
OUTPUT_DIR = "dataset/WLASL"
NSLT_JSON = "dataset/WLASL/nslt_100.json"
CLASS_LIST = "dataset/WLASL/wlasl_class_list.txt"


# ============================================================
# KEYPOINT CONFIGURATION
# ============================================================

TOTAL_KEYPOINTS = 75


# ============================================================
# LOAD WLASL-100 METADATA
# ============================================================

def load_metadata():

    with open(NSLT_JSON, "r", encoding="utf-8") as f:
        nslt_data = json.load(f)

    class_names = {}

    with open(CLASS_LIST, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                continue

            class_id = int(parts[0])
            class_name = parts[1].strip()

            class_names[class_id] = class_name

    return nslt_data, class_names


# ============================================================
# EXTRACT KEYPOINTS FROM ONE VIDEO
# ============================================================

def extract_keypoints(video_path):

    base_options = python.BaseOptions(
        model_bsset_path=MODEL_PATH
    )

    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    frames = []

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    with vision.HolisticLandmarker.create_from_options(
        options
    ) as landmarker:

        frame_index = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # ------------------------------------------------
            # BGR -> RGB
            # ------------------------------------------------

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            # ------------------------------------------------
            # VIDEO MODE TIMESTAMP
            # ------------------------------------------------

            timestamp_ms = int(
                frame_index * 1000 / fps
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # ------------------------------------------------
            # EMPTY 75 x 2 KEYPOINT ARRAY
            #
            # 0 - 32  = Pose
            # 33 - 53 = Left hand
            # 54 - 74 = Right hand
            # ------------------------------------------------

            keypoints = np.zeros(
                (TOTAL_KEYPOINTS, 2),
                dtype=np.float32
            )

            # ------------------------------------------------
            # POSE: 33 POINTS
            # ------------------------------------------------

            if result.pose_landmarks:

                for i, landmark in enumerate(
                    result.pose_landmarks
                ):

                    if i >= 33:
                        break

                    keypoints[i] = [
                        landmark.x,
                        landmark.y
                    ]

            # ------------------------------------------------
            # LEFT HAND: 21 POINTS
            # ------------------------------------------------

            if result.left_hand_landmarks:

                for i, landmark in enumerate(
                    result.left_hand_landmarks
                ):

                    if i >= 21:
                        break

                    keypoints[33 + i] = [
                        landmark.x,
                        landmark.y
                    ]

            # ------------------------------------------------
            # RIGHT HAND: 21 POINTS
            # ------------------------------------------------

            if result.right_hand_landmarks:

                for i, landmark in enumerate(
                    result.right_hand_landmarks
                ):

                    if i >= 21:
                        break

                    keypoints[54 + i] = [
                        landmark.x,
                        landmark.y
                    ]

            frames.append(keypoints)

            frame_index += 1

    cap.release()

    if not frames:

        raise RuntimeError(
            f"No frames extracted from: {video_path}"
        )

    return np.stack(frames)


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_name,
    nslt_data,
    class_names
):

    video_id = os.path.splitext(video_name)[0]

    # --------------------------------------------------------
    # CHECK METADATA
    # --------------------------------------------------------

    if video_id not in nslt_data:

        raise RuntimeError(
            f"{video_id} is not present in nslt_100.json"
        )

    metadata = nslt_data[video_id]

    subset = metadata["subset"]

    action = metadata["action"]

    class_id = action[0]

    if class_id not in class_names:

        raise RuntimeError(
            f"Class ID {class_id} not found in "
            f"wlasl_class_list.txt"
        )

    class_name = class_names[class_id]

    # --------------------------------------------------------
    # VIDEO PATH
    # --------------------------------------------------------

    video_path = os.path.join(
        VIDEO_DIR,
        video_name
    )

    if not os.path.exists(video_path):

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_dir = os.path.join(
        OUTPUT_DIR,
        subset,
        class_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # OUTPUT FILE
    # --------------------------------------------------------

    output_path = os.path.join(
        output_dir,
        video_name + ".pkl"
    )

    # --------------------------------------------------------
    # SKIP ALREADY PROCESSED VIDEO
    # --------------------------------------------------------

    if os.path.exists(output_path):

        print(
            f"Already processed: {output_path}"
        )

        return "skipped"

    # --------------------------------------------------------
    # PRINT METADATA
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(f"Video:      {video_name}")
    print(f"Subset:     {subset}")
    print(f"Class ID:   {class_id}")
    print(f"Class:      {class_name}")
    print("=" * 60)

    print(
        f"Processing: {video_path}"
    )

    # --------------------------------------------------------
    # EXTRACT KEYPOINTS
    # --------------------------------------------------------

    keypoints = extract_keypoints(
        video_path
    )

    print(
        "Extraction successful."
    )

    print(
        f"Keypoint shape: {keypoints.shape}"
    )

    print(
        "Expected:       (T, 75, 2)"
    )

    if keypoints.shape[1:] != (75, 2):

        raise RuntimeError(
            f"Unexpected keypoint shape: "
            f"{keypoints.shape}"
        )

    print(
        f"Number of frames: {keypoints.shape[0]}"
    )

    print(
        f"Min value: {keypoints.min():.4f}"
    )

    print(
        f"Max value: {keypoints.max():.4f}"
    )

    # --------------------------------------------------------
    # DATASET-COMPATIBLE SAMPLE
    # --------------------------------------------------------

    sample = {
        "keypoints": keypoints,
        "video": video_name,
        "class": class_name,
    }

    # --------------------------------------------------------
    # SAVE PICKLE
    # --------------------------------------------------------

    with open(
        output_path,
        "wb"
    ) as f:

        pickle.dump(
            sample,
            f
        )

    print(
        f"Saved: {output_path}"
    )

    return "processed"


# ============================================================
# PROCESS ALL WLASL-100 VIDEOS
# ============================================================

def process_all(
    nslt_data,
    class_names
):

    total = len(nslt_data)

    successful = 0
    skipped = 0
    failed = 0

    print()
    print("=" * 60)
    print("STARTING WLASL-100 PREPROCESSING")
    print("=" * 60)
    print(f"Total metadata entries: {total}")
    print("=" * 60)

    for i, video_id in enumerate(
        nslt_data.keys(),
        start=1
    ):

        video_name = f"{video_id}.mp4"

        print()
        print(
            f"[{i}/{total}] {video_name}"
        )

        try:

            result = process_video(
                video_name,
                nslt_data,
                class_names
            )

            if result == "processed":

                successful += 1

            elif result == "skipped":

                skipped += 1

        except Exception as e:

            failed += 1

            print(
                f"FAILED: {video_name}"
            )

            print(
                f"Reason: {e}"
            )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"Total:      {total}"
    )

    print(
        f"Processed:  {successful}"
    )

    print(
        f"Skipped:    {skipped}"
    )

    print(
        f"Failed:     {failed}"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract WLASL-100 keypoints and "
            "organize them for MODEL B."
        )
    )

    parser.add_argument(
        "--video",
        help=(
            "Process one video, "
            "e.g. 05238.mp4"
        )
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Process all WLASL-100 videos"
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # REQUIRE ONE MODE
    # --------------------------------------------------------

    if not args.video and not args.all:

        parser.error(
            "Use either --video VIDEO or --all"
        )

    if args.video and args.all:

        parser.error(
            "Use only one of --video or --all"
        )

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    print(
        "Loading WLASL-100 metadata..."
    )

    nslt_data, class_names = load_metadata()

    print(
        f"Loaded {len(nslt_data)} "
        f"WLASL-100 entries."
    )

    print(
        f"Loaded {len(class_names)} "
        f"class names."
    )

    # --------------------------------------------------------
    # SINGLE VIDEO
    # --------------------------------------------------------

    if args.video:

        process_video(
            args.video,
            nslt_data,
            class_names
        )

    # --------------------------------------------------------
    # ALL VIDEOS
    # --------------------------------------------------------

    elif args.all:

        process_all(
            nslt_data,
            class_names
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()