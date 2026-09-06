import numpy as np


def normalize_keypoints(keypoints):

    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    return keypoints