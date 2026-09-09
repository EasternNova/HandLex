import json
import os
import pickle

import numpy as np
import torch

from .augmentations import (
    rotate_keypoints,
    noise_injection,
    clip_frame,
    time_warp_uniform,
)


class Datasets(torch.utils.data.Dataset):
    """
    ModelB WLASL dataset.

    Expected structure:

        dataset/WLASL/
        ├── label2id.json
        ├── id2label.json
        ├── train/
        │   ├── class_name/
        │   │   ├── sample.pkl
        │   │   └── ...
        ├── val/
        └── test/

    Each sample must contain:
        {
            "keypoints": ...,
            "class": ...
        }

    Input keypoints:
        (T, 75, 2)

    ModelB selected joints:
        determined by joint_idxs

    Returned sample:
        keypoints: Tensor (T, joints, 2)
        label: Tensor ()
    """

    MAX_SEQUENCE_LENGTH = 64

    def __init__(
        self,
        root,
        split,
        shuffle=True,
        joint_idxs=None,
        augment=False,
    ):
        super().__init__()

        self.root = os.path.normpath(root)
        self.split = split
        self.joint_idxs = joint_idxs
        self.augment = augment

        # -------------------------------------------------
        # Validate dataset root
        # -------------------------------------------------

        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"Dataset root not found:\n{self.root}"
            )

        self.data_dir = os.path.join(
            self.root,
            split,
        )

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"Dataset split not found:\n{self.data_dir}"
            )

        # -------------------------------------------------
        # Load label mappings
        # -------------------------------------------------

        label2id_path = os.path.join(
            self.root,
            "label2id.json",
        )

        id2label_path = os.path.join(
            self.root,
            "id2label.json",
        )

        if not os.path.isfile(label2id_path):
            raise FileNotFoundError(
                f"Missing label2id.json:\n{label2id_path}"
            )

        if not os.path.isfile(id2label_path):
            raise FileNotFoundError(
                f"Missing id2label.json:\n{id2label_path}"
            )

        with open(label2id_path, "r", encoding="utf-8") as f:
            self.label2id = json.load(f)

        with open(id2label_path, "r", encoding="utf-8") as f:
            self.id2label = json.load(f)

        if not self.label2id:
            raise ValueError(
                "label2id.json contains no labels."
            )

        # -------------------------------------------------
        # Build sample list
        # -------------------------------------------------

        self.list_key = []

        for class_name in sorted(os.listdir(self.data_dir)):

            class_dir = os.path.join(
                self.data_dir,
                class_name,
            )

            if not os.path.isdir(class_dir):
                continue

            # Make sure every dataset class has a label.
            if class_name not in self.label2id:
                raise ValueError(
                    f"Class '{class_name}' is missing "
                    f"from label2id.json."
                )

            for sample_name in sorted(
                os.listdir(class_dir)
            ):

                sample_path = os.path.join(
                    class_dir,
                    sample_name,
                )

                if os.path.isfile(sample_path):
                    self.list_key.append(
                        sample_path
                    )

        if not self.list_key:
            raise ValueError(
                f"No samples found in:\n{self.data_dir}"
            )

        if shuffle:
            np.random.shuffle(self.list_key)

    # =====================================================
    # SAMPLE LOADING
    # =====================================================

    def __getitem__(self, i):

        key = self.list_key[i]

        # -------------------------------------------------
        # Load sample
        # -------------------------------------------------

        try:
            with open(key, "rb") as f:
                sample = pickle.load(f)

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load sample:\n{key}"
            ) from exc

        if not isinstance(sample, dict):
            raise ValueError(
                f"Invalid sample format:\n{key}"
            )

        if "keypoints" not in sample:
            raise KeyError(
                f"Sample is missing 'keypoints':\n{key}"
            )

        if "class" not in sample:
            raise KeyError(
                f"Sample is missing 'class':\n{key}"
            )

        # -------------------------------------------------
        # Extract x/y coordinates
        # -------------------------------------------------

        keypoints = np.asarray(
            sample["keypoints"],
            dtype=np.float32,
        )

        if keypoints.ndim != 3:
            raise ValueError(
                f"Expected keypoints with shape "
                f"(T, J, C), got {keypoints.shape} "
                f"in:\n{key}"
            )

        if keypoints.shape[-1] < 2:
            raise ValueError(
                f"Keypoints must contain x/y coordinates. "
                f"Got shape {keypoints.shape} in:\n{key}"
            )

        keypoints = keypoints[:, :, :2]

        # -------------------------------------------------
        # Verify directory class
        # -------------------------------------------------

        class_name = os.path.basename(
            os.path.dirname(key)
        )

        if sample["class"] != class_name:
            raise ValueError(
                f"Dataset class mismatch:\n"
                f"Directory: {class_name}\n"
                f"Sample:    {sample['class']}\n"
                f"File:      {key}"
            )

        # -------------------------------------------------
        # Convert class to numerical label
        # -------------------------------------------------

        try:
            label = self.label2id[
                sample["class"]
            ]
        except KeyError as exc:
            raise ValueError(
                f"Class '{sample['class']}' does not "
                f"exist in label2id.json."
            ) from exc

        # -------------------------------------------------
        # Validate sequence
        # -------------------------------------------------

        if keypoints.shape[0] == 0:
            raise ValueError(
                f"Empty keypoint sequence:\n{key}"
            )

        # -------------------------------------------------
        # Keep coordinates inside [0, 1]
        # -------------------------------------------------

        keypoints = np.clip(
            keypoints,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Maximum sequence length
        # -------------------------------------------------

        if keypoints.shape[0] > self.MAX_SEQUENCE_LENGTH:
            keypoints = clip_frame(
                keypoints,
                self.MAX_SEQUENCE_LENGTH,
                True,
            )

        # -------------------------------------------------
        # Training augmentation
        # -------------------------------------------------

        if (
            self.augment
            and np.random.uniform(0, 1) < 0.7
        ):
            keypoints = self.apply_augment(
                keypoints
            )

        # -------------------------------------------------
        # Guarantee maximum sequence length
        # -------------------------------------------------

        if keypoints.shape[0] > self.MAX_SEQUENCE_LENGTH:
            keypoints = clip_frame(
                keypoints,
                self.MAX_SEQUENCE_LENGTH,
                True,
            )

        # -------------------------------------------------
        # Normalize
        # -------------------------------------------------

        keypoints = self.normalize_keypoints(
            keypoints
        )

        # -------------------------------------------------
        # Final numerical validation
        # -------------------------------------------------

        if not np.all(np.isfinite(keypoints)):
            raise ValueError(
                f"NaN or Inf detected in keypoints:\n{key}"
            )

        # -------------------------------------------------
        # Convert to tensors
        # -------------------------------------------------

        keypoints = torch.from_numpy(
            keypoints
        ).float()

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        return keypoints, label

    # =====================================================
    # AUGMENTATION
    # =====================================================

    def apply_augment(self, keypoints):
        """
        Apply mild augmentation while preserving
        the overall sign structure.
        """

        # -------------------------------------------------
        # 1. Mild rotation
        # -------------------------------------------------

        if np.random.uniform(0, 1) < 0.5:

            angle = np.random.uniform(
                -8,
                8,
            )

            keypoints = rotate_keypoints(
                keypoints,
                (0.5, 0.5),
                angle,
            )

        # -------------------------------------------------
        # 2. Small coordinate noise
        # -------------------------------------------------

        if np.random.uniform(0, 1) < 0.5:

            noise_level = np.random.uniform(
                0.005,
                0.03,
            )

            keypoints = noise_injection(
                keypoints,
                noise_level,
            )

        # -------------------------------------------------
        # 3. Mild temporal variation
        # -------------------------------------------------

        if np.random.uniform(0, 1) < 0.3:

            n_frames = keypoints.shape[0]

            if n_frames > 40:

                minimum = max(
                    40,
                    n_frames - 12,
                )

                target = np.random.randint(
                    minimum,
                    n_frames + 1,
                )

                keypoints = clip_frame(
                    keypoints,
                    target,
                    True,
                )

        # -------------------------------------------------
        # 4. Mild speed variation
        # -------------------------------------------------

        if np.random.uniform(0, 1) < 0.3:

            speed = np.random.uniform(
                0.9,
                1.1,
            )

            keypoints = time_warp_uniform(
                keypoints,
                speed,
            )

        # -------------------------------------------------
        # Keep coordinates valid
        # -------------------------------------------------

        keypoints = np.clip(
            keypoints,
            0.0,
            1.0,
        )

        return keypoints

    # =====================================================
    # CLASS QUERY
    # =====================================================

    def query_class(
        self,
        class_name,
        _max=5,
    ):
        """
        Return up to _max samples belonging
        to exactly the requested class.
        """

        matching_indices = []

        for index, path in enumerate(
            self.list_key
        ):

            path_class = os.path.basename(
                os.path.dirname(path)
            )

            if path_class == class_name:
                matching_indices.append(index)

            if len(matching_indices) >= _max:
                break

        return [
            self[i]
            for i in matching_indices
        ]

    # =====================================================
    # LENGTH
    # =====================================================

    def __len__(self):
        return len(self.list_key)

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def normalize_keypoints(self, keypoints):
        """
        Normalize selected joints independently
        for every frame.

        NOTE:
        This preserves the current ModelB normalization
        strategy. It should be treated as a separate
        experiment later rather than changed silently.
        """

        if self.joint_idxs is None:
            return keypoints

        for i in range(keypoints.shape[0]):

            keypoints[
                i,
                self.joint_idxs,
                :,
            ] = self._normalize_part(
                keypoints[
                    i,
                    self.joint_idxs,
                    :,
                ]
            )

        return keypoints

    @staticmethod
    def _normalize_part(keypoint):
        """
        Normalize one group of keypoints while
        preserving relative geometry.
        """

        if keypoint.shape[-1] != 2:
            raise ValueError(
                "Keypoints must have x and y coordinates."
            )

        x_coords = keypoint[:, 0]
        y_coords = keypoint[:, 1]

        min_x = np.min(x_coords)
        min_y = np.min(y_coords)

        max_x = np.max(x_coords)
        max_y = np.max(y_coords)

        width = max_x - min_x
        height = max_y - min_y

        # -------------------------------------------------
        # Make normalization region square
        # -------------------------------------------------

        if width > height:

            delta_x = 0.05 * width
            delta_y = (
                delta_x
                + ((width - height) / 2)
            )

        else:

            delta_y = 0.05 * height
            delta_x = (
                delta_y
                + ((height - width) / 2)
            )

        start_point = [
            max(
                0.0,
                min(
                    min_x - delta_x,
                    1.0,
                ),
            ),
            max(
                0.0,
                min(
                    min_y - delta_y,
                    1.0,
                ),
            ),
        ]

        end_point = [
            max(
                0.0,
                min(
                    max_x + delta_x,
                    1.0,
                ),
            ),
            max(
                0.0,
                min(
                    max_y + delta_y,
                    1.0,
                ),
            ),
        ]

        # -------------------------------------------------
        # Normalize X
        # -------------------------------------------------

        region_width = (
            end_point[0]
            - start_point[0]
        )

        if region_width > 0.0:

            keypoint[:, 0] = (
                keypoint[:, 0]
                - start_point[0]
            ) / region_width

        # -------------------------------------------------
        # Normalize Y
        # -------------------------------------------------

        region_height = (
            end_point[1]
            - start_point[1]
        )

        if region_height > 0.0:

            keypoint[:, 1] = (
                keypoint[:, 1]
                - start_point[1]
            ) / region_height

        return keypoint

    # =====================================================
    # PADDING
    # =====================================================

    def padding_keypoint(
        self,
        keypoint,
        max_len,
    ):
        """
        Pad a sequence with zero frames.
        """

        T, K, C = keypoint.shape

        if T >= max_len:
            return keypoint

        padding = torch.zeros(
            (
                max_len - T,
                K,
                C,
            ),
            dtype=keypoint.dtype,
        )

        return torch.cat(
            [
                keypoint,
                padding,
            ],
            dim=0,
        )

    # =====================================================
    # COLLATOR
    # =====================================================

    def data_collator(self, batch):
        """
        Collate variable-length sequences.

        Returns:
            keypoints:
                (batch, time, joints, 2)

            attention_mask:
                (batch, time)

            labels:
                (batch,)
        """

        if not batch:
            raise ValueError(
                "Cannot collate an empty batch."
            )

        keypoints = [
            item[0]
            for item in batch
        ]

        labels = [
            item[1]
            for item in batch
        ]

        lengths = [
            x.shape[0]
            for x in keypoints
        ]

        max_len = max(lengths)

        attention_masks = []
        padded_keypoints = []

        for keypoint in keypoints:

            T = keypoint.shape[0]

            if T < max_len:

                mask = torch.cat(
                    [
                        torch.ones(
                            T,
                            dtype=torch.float32,
                        ),
                        torch.zeros(
                            max_len - T,
                            dtype=torch.float32,
                        ),
                    ],
                    dim=0,
                )

                padded = self.padding_keypoint(
                    keypoint,
                    max_len,
                )

            else:

                mask = torch.ones(
                    T,
                    dtype=torch.float32,
                )

                padded = keypoint

            attention_masks.append(mask)
            padded_keypoints.append(padded)

        keypoints = torch.stack(
            padded_keypoints
        )

        attention_mask = torch.stack(
            attention_masks
        )

        labels = torch.stack(
            labels
        )

        return {
            "keypoints": keypoints,
            "attention_mask": attention_mask,
            "labels": labels,
        }