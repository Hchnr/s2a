"""Offline right-local SE3 reference error decomposition, never an online policy."""

import numpy as np

from s2a.geometry import finite, rot6d_to_matrix


def decompose(states, actions):
    """Separate native-frame local increment errors from orientation coupling.

    Truth is privileged offline evidence. For every step:
      R_pred*d_pred - dp_truth
        = (R_truth*d_pred - dp_truth) + (R_pred-R_truth)*d_pred.
    Cumulative terms sum to the actual integrated position error. Their squared
    norms are not independent contributions: the signed cross term is required.
    """
    actions, states = finite(actions), finite(states)
    if actions.ndim != 2 or actions.shape[1] != 128 or states.shape != (len(actions) + 1, 128) or len(actions) == 0:
        raise ValueError("Complete measured endpoints and full128 local increments required")
    if actions[:, 10:].any():
        raise ValueError("Inactive action slots must be zero")
    truth_rotation = rot6d_to_matrix(states[:, 3:9])
    increments = rot6d_to_matrix(actions[:, 3:9])
    orientation, position = truth_rotation[0].copy(), states[0, :3].copy()
    local, frame, endpoint = [], [], []
    truth_world = np.diff(states[:, :3], axis=0)
    for index, action in enumerate(actions):
        local.append(truth_rotation[index] @ action[:3] - truth_world[index])
        frame.append((orientation - truth_rotation[index]) @ action[:3])
        position += orientation @ action[:3]
        orientation = orientation @ increments[index]
        endpoint.append(position - states[index + 1, :3])
    local = np.asarray(local)
    frame = np.asarray(frame)
    cumulative_local, cumulative_frame = np.cumsum(local, axis=0), np.cumsum(frame, axis=0)
    endpoint = np.asarray(endpoint)
    if not np.allclose(endpoint, cumulative_local + cumulative_frame, rtol=0, atol=1e-12):
        raise ValueError("Integrated vector decomposition identity failed")
    return dict(
        local_world_increment_error=local,
        frame_world_increment_error=frame,
        local_cumulative=cumulative_local,
        frame_cumulative=cumulative_frame,
        total_cumulative=endpoint,
        opening_error=np.clip(actions[:, 9], 0, 1) - states[1:, 9],
        predicted_opening=np.clip(actions[:, 9], 0, 1),
        truth_opening=states[1:, 9].copy(),
    )


def position_summary(parts):
    local, frame, total = (
        np.asarray(parts[key]) for key in ("local_cumulative", "frame_cumulative", "total_cumulative")
    )
    local_mse, frame_mse, total_mse = (float(np.mean(np.sum(v * v, axis=-1))) for v in (local, frame, total))
    cross = float(2 * np.mean(np.sum(local * frame, axis=-1)))
    if abs(total_mse - local_mse - frame_mse - cross) > 1e-12:
        raise ValueError("Squared error decomposition failed")
    return dict(
        total_mse_m2=total_mse,
        local_cumulative_mse_m2=local_mse,
        frame_cumulative_mse_m2=frame_mse,
        cross_m2=cross,
        local_increment_mse_m2=float(np.mean(np.sum(np.asarray(parts["local_world_increment_error"]) ** 2, axis=-1))),
    )
