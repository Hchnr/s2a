"""Small fixed-interface gripper inverse features and reachable action choice."""

import numpy as np

OFFSETS = (0, 1, 3, 7)


def raw_features(opening, motor_target, previous_opening_delta, generated_openings):
    reference = np.asarray(generated_openings, dtype=float)
    state = np.asarray([opening, motor_target, previous_opening_delta], dtype=float)
    if reference.ndim != 1 or not len(reference) or not np.isfinite(reference).all():
        raise ValueError("Nonempty already generated scalar opening reference required")
    if not np.isfinite(state).all() or not 0 <= opening <= 1 or not -1 <= motor_target <= 1:
        raise ValueError("Invalid current gripper state")
    if np.any((reference < 0) | (reference > 1)):
        raise ValueError("Opening reference must lie in [0,1]")
    values = [float(reference[min(i, len(reference) - 1)]) - opening for i in OFFSETS]
    available = np.array([i < len(reference) for i in OFFSETS], dtype=bool)
    return np.r_[state, values], available


def design_features(raw, available, mean, scale):
    raw = np.asarray(raw, dtype=float)
    available = np.asarray(available)
    mean, scale = np.asarray(mean, dtype=float), np.asarray(scale, dtype=float)
    if raw.shape[-1:] != (7,) or mean.shape != (7,) or scale.shape != (7,):
        raise ValueError("Seven continuous state/reference features required")
    if available.shape != raw.shape[:-1] + (4,) or available.dtype != bool:
        raise ValueError("Four explicit reference availability bits required")
    if not all(np.isfinite(a).all() for a in (raw, mean, scale)) or np.any(scale <= 0):
        raise ValueError("Finite fit normalization with positive scale required")
    z = (raw - mean) / scale
    quadratic = np.stack([z[..., i] * z[..., j] for i in range(7) for j in range(i, 7)], axis=-1)
    return np.concatenate([z, quadratic, available.astype(float), np.ones(raw.shape[:-1] + (1,))], axis=-1)


def reachable_command(motor_target, predicted_motor_target):
    if not np.isfinite([motor_target, predicted_motor_target]).all() or not -1 <= motor_target <= 1:
        raise ValueError("Finite normalized current and predicted motor targets required")
    prediction = float(np.clip(predicted_motor_target, -1, 1))
    commands = np.array([0.0, -1.0, 1.0])
    reachable = np.clip(motor_target - 0.25 * commands, -1, 1)
    chosen = int(np.argmin(np.square(reachable - prediction)))
    return dict(
        command=float(commands[chosen]),
        predicted_motor_target=prediction,
        selected_motor_target=float(reachable[chosen]),
        reachable_motor_targets=reachable.tolist(),
    )
