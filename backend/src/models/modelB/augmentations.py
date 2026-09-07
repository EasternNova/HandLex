import numpy as np


def rotate_keypoints(frames, origin, angle_degrees):
    """
    Rotate all keypoints around the given origin.
    """

    angle_radians = np.radians(angle_degrees)

    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)

    rotation_matrix = np.array([
        [cos_angle, -sin_angle],
        [sin_angle,  cos_angle]
    ])

    shifted_points = frames - np.array(origin)

    rotated_points = np.einsum(
        "ij,klj->kli",
        rotation_matrix,
        shifted_points
    )

    return rotated_points + np.array(origin)


def noise_injection(frames, noise_level):
    """
    Add small Gaussian noise to keypoints.

    Keep this small because coordinates are normalized to [0, 1].
    """

    noise = np.random.normal(
        0,
        noise_level,
        frames.shape
    )

    noisy_frames = frames + noise

    return np.clip(noisy_frames, 0.0, 1.0)


def clip_frame(frames, tgt_frame, is_uniform=True):
    """
    Temporally sample a sequence to a target number of frames.
    """

    t = frames.shape[0]

    if t == 0 or tgt_frame <= 0:
        return frames

    if tgt_frame >= t:
        return frames

    if is_uniform:
        indices = np.linspace(
            0,
            t - 1,
            tgt_frame
        ).astype(int)

    else:
        indices = np.sort(
            np.random.choice(
                np.arange(t),
                size=tgt_frame,
                replace=False
            )
        )

    return frames[indices]


def time_warp_uniform(frames, scale):
    """
    Mild temporal speed variation.

    scale > 1  -> slightly faster
    scale < 1  -> slightly slower
    """

    t = frames.shape[0]

    if t <= 1:
        return frames

    new_t = max(2, int(t * scale))

    indices = np.linspace(
        0,
        t - 1,
        new_t
    ).astype(int)

    indices = np.clip(
        indices,
        0,
        t - 1
    )

    return frames[indices]


def flip_keypoints(keypoints):
    """
    Horizontal flip of normalized x coordinates.
    """

    flipped_keypoints = keypoints.copy()

    flipped_keypoints[..., 0] = (
        1.0 - flipped_keypoints[..., 0]
    )

    return flipped_keypoints