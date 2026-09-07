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

    def __init__(
        self,
        root,
        split,
        shuffle=True,
        joint_idxs=None,
        augment=False,
    ):
        self.root = root
        self.split = split
        self.joint_idxs = joint_idxs
        self.augment = augment

        # Load label mappings
        with open(
            f"{self.root}/label2id.json",
            "r"
        ) as f:
            self.label2id = json.load(f)

        with open(
            f"{self.root}/id2label.json",
            "r"
        ) as f:
            self.id2label = json.load(f)

        # Dataset directory
        self.data_dir = f"{root}/{split}"

        # Build list of all samples
        self.list_key = [
            f"{self.data_dir}/{class_name}/{sample_name}"
            for class_name in os.listdir(self.data_dir)
            for sample_name in os.listdir(
                f"{self.data_dir}/{class_name}"
            )
        ]

        if shuffle:
            np.random.shuffle(self.list_key)

    def __getitem__(self, i):

        key = self.list_key[i]

        # Load sample
        with open(key, "rb") as f:
            sample = pickle.load(f)

        # Extract x/y coordinates only
        keypoints = np.array(
            sample["keypoints"]
        )[:, :, :2]

        # Make sure dataset folder and stored class agree
        class_name = os.path.basename(
            os.path.dirname(key)
        )

        assert (
            sample["class"] == class_name
        ), (
            f"{sample['class']} != {class_name}"
        )

        # Convert class name to numerical label
        label = self.label2id[
            sample["class"]
        ]

        # Keep coordinates inside [0, 1]
        keypoints = np.clip(
            keypoints,
            0.0,
            1.0
        )

        # Maximum sequence length = 64
        if keypoints.shape[0] > 64:
            keypoints = clip_frame(
                keypoints,
                64,
                True
            )

        # Apply augmentation only to training data
        if (
            self.augment
            and np.random.uniform(0, 1) < 0.7
        ):
            keypoints = self.apply_augment(
                keypoints
            )

        # Normalize selected joints
        keypoints = self.normalize_keypoints(
            keypoints
        )

        # Convert to PyTorch tensors
        keypoints = torch.from_numpy(
            keypoints
        ).float()

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return keypoints, label

    def apply_augment(self, keypoints):
        """
        Apply mild augmentation to improve
        generalization without destroying
        the sign structure.
        """

        # -------------------------------------------------
        # 1. Mild rotation
        # -------------------------------------------------
        if np.random.uniform(0, 1) < 0.5:

            angle = np.random.uniform(
                -8,
                8
            )

            # Rotate around image center
            keypoints = rotate_keypoints(
                keypoints,
                (0.5, 0.5),
                angle
            )

        # -------------------------------------------------
        # 2. Small coordinate noise
        # -------------------------------------------------
        if np.random.uniform(0, 1) < 0.5:

            noise_level = np.random.uniform(
                0.005,
                0.03
            )

            keypoints = noise_injection(
                keypoints,
                noise_level
            )

        # -------------------------------------------------
        # 3. Mild temporal variation
        # -------------------------------------------------
        if np.random.uniform(0, 1) < 0.3:

            n_f = keypoints.shape[0]

            if n_f > 40:

                target = np.random.randint(
                    max(40, n_f - 12),
                    n_f + 1
                )

                keypoints = clip_frame(
                    keypoints,
                    target,
                    True
                )

        # -------------------------------------------------
        # 4. Mild speed variation
        # -------------------------------------------------
        if np.random.uniform(0, 1) < 0.3:

            speed = np.random.uniform(
                0.9,
                1.1
            )

            keypoints = time_warp_uniform(
                keypoints,
                speed
            )

        # Keep coordinates valid
        keypoints = np.clip(
            keypoints,
            0.0,
            1.0
        )

        return keypoints

    def query_class(
        self,
        class_name,
        _max=5
    ):
        """
        Return up to _max samples belonging
        to a specific class.
        """

        key_queries = [
            self.list_key.index(x)
            for x in self.list_key
            if class_name in x
        ][:_max]

        return [
            self[i]
            for i in key_queries
        ]

    def __len__(self):
        return len(self.list_key)

    def normalize_keypoints(self, keypoints):
        """
        Normalize the selected joints independently
        for every frame.
        """

        if self.joint_idxs is None:
            return keypoints

        for i in range(keypoints.shape[0]):

            keypoints[
                i,
                self.joint_idxs,
                :
            ] = self._normalize_part(
                keypoints[
                    i,
                    self.joint_idxs,
                    :
                ]
            )

        return keypoints

    @staticmethod
    def _normalize_part(keypoint):
        """
        Normalize a group of keypoints while
        preserving their relative geometry.
        """

        assert (
            keypoint.shape[-1] == 2
        ), "Keypoints must have x, y"

        x_coords = keypoint[:, 0]
        y_coords = keypoint[:, 1]

        min_x = np.min(x_coords)
        min_y = np.min(y_coords)

        max_x = np.max(x_coords)
        max_y = np.max(y_coords)

        width = max_x - min_x
        height = max_y - min_y

        # Add a small margin and make the
        # normalization square
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
                    1.0
                )
            ),
            max(
                0.0,
                min(
                    min_y - delta_y,
                    1.0
                )
            ),
        ]

        end_point = [
            max(
                0.0,
                min(
                    max_x + delta_x,
                    1.0
                )
            ),
            max(
                0.0,
                min(
                    max_y + delta_y,
                    1.0
                )
            ),
        ]

        # Normalize X
        width = (
            end_point[0]
            - start_point[0]
        )

        if width != 0.0:

            keypoint[:, 0] = (
                keypoint[:, 0]
                - start_point[0]
            ) / width

        # Normalize Y
        height = (
            end_point[1]
            - start_point[1]
        )

        if height != 0.0:

            keypoint[:, 1] = (
                keypoint[:, 1]
                - start_point[1]
            ) / height

        return keypoint

    def padding_keypoint(
        self,
        keypoint,
        max_len
    ):
        """
        Pad a sequence with zero frames.
        """

        T, K, C = keypoint.shape

        padding = torch.zeros(
            (
                max_len - T,
                K,
                C
            ),
            dtype=keypoint.dtype
        )

        return torch.cat(
            [
                keypoint,
                padding
            ],
            dim=0
        )

    def data_collator(self, batch):
        """
        Collate variable-length sequences into
        a padded batch with attention masks.
        """

        keypoints = [
            x[0]
            for x in batch
        ]

        labels = [
            x[1]
            for x in batch
        ]

        lengths = [
            x.shape[0]
            for x in keypoints
        ]

        max_len = max(lengths)

        attention_masks = []
        keypoint_paddings = []

        for keypoint in keypoints:

            T, K, C = keypoint.shape

            if T < max_len:

                mask = torch.cat(
                    [
                        torch.ones(T),
                        torch.zeros(
                            max_len - T
                        ),
                    ],
                    dim=0
                )

                kp_padd = self.padding_keypoint(
                    keypoint,
                    max_len
                )

            else:

                mask = torch.ones(T)
                kp_padd = keypoint

            attention_masks.append(mask)
            keypoint_paddings.append(
                kp_padd
            )

        keypoints = torch.stack(
            keypoint_paddings
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