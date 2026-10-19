"""Causal state baselines and explicit phase proxies for checkpoint evaluation."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, pose_error, validate_transform


def extrapolate_history(history, horizon, history_steps=4):
    """Constant world velocity using only poses available through the anchor."""
    history = validate_transform(history)
    if history.ndim != 3 or not len(history) or horizon < 1 or history_steps < 1:
        raise ValueError("requires observed history and positive horizon/history_steps")
    anchor = history[-1]
    count = min(history_steps, len(history) - 1)
    future = np.repeat(anchor[None], horizon, axis=0)
    if count:
        previous = history[-1 - count]
        k = np.arange(1, horizon + 1) / count
        future[:, :3, 3] += k[:, None] * (anchor[:3, 3] - previous[:3, 3])
        angular = Rotation.from_matrix(anchor[:3, :3] @ previous[:3, :3].T).as_rotvec()
        future[:, :3, :3] = Rotation.from_rotvec(k[:, None] * angular).as_matrix() @ anchor[:3, :3]
    return future


def phase_anchors(states):
    """Select observed gripper/height events; proxies do not assert object grasp."""
    states = finite(states)
    if states.ndim != 2 or states.shape[1] < 10 or len(states) < 20:
        raise ValueError("requires a complete robot state episode")
    opening = states[:, 9]
    closed = np.flatnonzero(opening < 0.35)
    close = int(closed[0]) if len(closed) else len(states) // 3
    release_candidates = np.flatnonzero((np.arange(len(states)) > close + 3) & (opening > 0.65))
    release = int(release_candidates[0]) if len(release_candidates) else 3 * len(states) // 4
    stop = max(close + 1, release)
    transport = close + int(np.argmax(states[close:stop, 2]))
    events = [
        ("approach", 0, True),
        ("closing_proxy", close, bool(len(closed))),
        ("transport_height_proxy", transport, bool(len(closed))),
        ("release_proxy", release, bool(len(release_candidates) and len(closed))),
    ]
    return [
        dict(
            phase=name,
            event_frame=event,
            anchor_frame=min(max(0, event - (8 if name != "approach" else 0)), len(states) - 17),
            event_detected=detected,
            interpretation="robot-state proxy; no object-contact or task-success claim",
        )
        for name, event, detected in events
    ]


def state_metrics(reference, predicted, true_opening, predicted_opening):
    """Evaluate aligned valid future states; the caller excludes padded endpoints."""
    error = pose_error(reference, predicted)
    grip = finite(predicted_opening) - finite(true_opening)
    if error.ndim != 2 or not len(error) or grip.shape != (len(error),):
        raise ValueError("one opening per valid endpoint is required")
    position, rotation = np.linalg.norm(error[:, :3], axis=1), np.linalg.norm(error[:, 3:], axis=1)
    return dict(
        endpoints=len(error),
        position_rmse_m=float(np.sqrt(np.mean(position**2))),
        rotation_rmse_rad=float(np.sqrt(np.mean(rotation**2))),
        opening_mae=float(np.mean(abs(grip))),
        position_errors_m=position.tolist(),
        rotation_errors_rad=rotation.tolist(),
        opening_errors=grip.tolist(),
    )
