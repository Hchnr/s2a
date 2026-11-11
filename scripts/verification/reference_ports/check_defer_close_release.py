"""Independent scalar reconstruction of a close-deferral proposal."""

import numpy as np
from scipy.spatial.transform import Rotation


def independent_proposal(poses, openings, index, measured_tcp, measured_opening, motor_target, baseline, closing):
    poses, openings, measured, motor = (
        np.asarray(poses, dtype=float),
        np.asarray(openings, dtype=float),
        np.asarray(measured_tcp, dtype=float),
        np.asarray(motor_target, dtype=float),
    )
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or openings.shape != (len(poses),)
        or measured.shape != (4, 4)
        or motor.shape != (2,)
        or type(index) is not int
        or not 0 <= index < len(openings)
        or type(closing) is not bool
        or not all(np.isfinite(x).all() for x in (poses, openings, measured, motor))
        or not 0 <= measured_opening <= 1
        or not -1 <= baseline <= 1
        or np.any((openings < 0) | (openings > 1))
        or np.any(abs(motor) > 1)
        or abs(float(motor.sum())) > 1e-10
    ):
        raise ValueError("independent close-deferral inputs differ")
    stop = min(index + 6, len(openings) - 1)
    advances = bool(
        all(openings[i + 1] >= openings[i] - 1e-12 for i in range(index, stop)) and openings[stop] > openings[index]
    )
    selected = stop if advances else index
    anchor = poses[index]
    segment = poses[index : selected + 1]
    position_span = max(float(np.linalg.norm(p[:3, 3] - anchor[:3, 3])) for p in segment)
    rotation_span = max(float(Rotation.from_matrix(anchor[:3, :3].T @ p[:3, :3]).magnitude()) for p in segment)
    position_error = float(np.linalg.norm(measured[:3, 3] - anchor[:3, 3]))
    rotation_error = float(Rotation.from_matrix(anchor[:3, :3].T @ measured[:3, :3]).magnitude())
    guards = dict(
        position_span=position_span <= 0.01,
        rotation_span=rotation_span <= 0.1,
        measured_position=position_error <= 0.01,
        measured_rotation=rotation_error <= 0.05,
    )
    motor_opening = float(np.clip((0.039 + 0.02 * (motor[0] - motor[1])) / 0.079, 0, 1))
    rise = float(openings[selected] - openings[index])
    deferred = bool(
        baseline == 1.0
        and closing
        and motor_opening <= measured_opening
        and index + 6 < len(openings)
        and advances
        and rise >= 0.01 / 0.079
        and not guards["position_span"]
        and guards["rotation_span"]
        and guards["measured_position"]
        and guards["measured_rotation"]
    )
    return dict(
        command=0.0 if deferred else float(baseline),
        deferred=deferred,
        motor_opening=motor_opening,
        predicted_rise=rise,
        guards=guards,
        proposal_advanced=advances,
        proposal_selected_index=selected,
        position_span=position_span,
        rotation_span=rotation_span,
        measured_position=position_error,
        measured_rotation=rotation_error,
    )


def verify_proposal(candidate, checked):
    preview = candidate["preview"]
    original = preview["preview_proposal"]
    if (
        candidate["command"] != checked["command"]
        or candidate["deferred"] != checked["deferred"]
        or preview["guards"] != checked["guards"]
        or original["advanced"] != checked["proposal_advanced"]
        or original["selected_index"] != checked["proposal_selected_index"]
    ):
        raise ValueError("independent close-deferral decision differs")
    pairs = (
        (candidate["motor_opening"], checked["motor_opening"]),
        (candidate["predicted_rise"], checked["predicted_rise"]),
        (preview["position_span_m"], checked["position_span"]),
        (preview["rotation_span_rad"], checked["rotation_span"]),
        (preview["measured_position_error_m"], checked["measured_position"]),
        (preview["measured_rotation_error_rad"], checked["measured_rotation"]),
    )
    maximum = max(abs(float(a) - float(b)) for a, b in pairs)
    if maximum > 1e-10:
        raise ValueError("independent close-deferral geometry differs")
    return float(maximum)
