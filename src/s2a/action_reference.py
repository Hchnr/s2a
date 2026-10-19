"""One physical Action128 endpoint decoder for WAM and privileged controls."""

import numpy as np

from s2a.geometry import finite, integrate_local, inverse, matrix_to_rot6d, validate_transform


def decode_action_targets(anchor_tcp, actions, *, horizon):
    actions = finite(actions)
    if actions.ndim != 2 or actions.shape[1] != 128 or np.any(actions[:, 10:] != 0):
        raise ValueError("requires physical Action128 with ten active slots")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= len(actions):
        raise ValueError("invalid execution horizon")
    poses = integrate_local(anchor_tcp, actions[:horizon, :9])[1:]
    return poses, np.clip(actions[:horizon, 9], 0, 1)


def encode_oracle_targets(anchor_tcp, target_tcp, openings):
    """Privileged oracle only; no expert commands and no physical-state restore.

    First delta connects the current measured anchor to the first supplied
    world endpoint. Later deltas connect consecutive already-generated targets.
    """
    anchor = validate_transform(finite(anchor_tcp, (4, 4)))
    poses = validate_transform(target_tcp)
    openings = finite(openings)
    if poses.ndim != 3 or len(poses) == 0 or openings.shape != (len(poses),):
        raise ValueError("requires nonempty endpoints and one opening per pose")
    if np.any((openings < 0) | (openings > 1)):
        raise ValueError("oracle openings must satisfy the published range")
    previous = np.concatenate([anchor[None], poses[:-1]])
    delta = inverse(previous) @ poses
    result = np.zeros((len(poses), 128))
    result[:, :3], result[:, 3:9], result[:, 9] = delta[:, :3, 3], matrix_to_rot6d(delta[:, :3, :3]), openings
    return result
