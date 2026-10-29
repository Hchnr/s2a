"""Finite-difference response measurements; no inverse fitting or task claims."""

import numpy as np
from scipy.spatial.transform import Rotation


def endpoint(packet):
    pose = np.asarray(packet["pose"], dtype=float)
    twist = np.asarray(packet["twist"], dtype=float)
    command = np.asarray(packet["command"], dtype=float)
    if pose.shape != (4, 4) or twist.shape != (6,) or command.shape != (7,):
        raise ValueError("Physical site pose, world twist, and native7 command required")
    if not all(np.isfinite(x).all() for x in (pose, twist, command)):
        raise ValueError("Finite physical measurements required")
    if (
        not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), rtol=0, atol=1e-8)
        or np.linalg.det(pose[:3, :3]) < 0
    ):
        raise ValueError("Rigid physical pose required")
    return pose, twist, command


def difference(left, right):
    pl, vl, _ = left
    pr, vr, _ = right
    return dict(
        position=pr[:3, 3] - pl[:3, 3],
        linear_velocity=vr[:3] - vl[:3],
        so3_increment=Rotation.from_matrix(pr[:3, :3] @ pl[:3, :3].T).as_rotvec(),
        angular_velocity=vr[3:] - vl[3:],
    )


def axis_response(baseline, minus, plus, axis, physical_delta_per_input):
    if type(axis) is not int or axis not in (0, 1, 2):
        raise ValueError("Translation axis required")
    scale = float(physical_delta_per_input)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Positive actual interface scale required")
    base, negative, positive = endpoint(baseline), endpoint(minus), endpoint(plus)
    other = [i for i in range(7) if i != axis]
    if not np.array_equal(base[2][other], negative[2][other]) or not np.array_equal(base[2][other], positive[2][other]):
        raise ValueError("Only the registered command axis may change")
    if any(np.any(np.abs(x[2]) > 1) for x in (base, negative, positive)):
        raise ValueError("Actual native command interface exceeded")
    dm = float(negative[2][axis] - base[2][axis])
    dp = float(positive[2][axis] - base[2][axis])
    if not dm <= 0 <= dp or dp - dm <= 0:
        raise ValueError("Actual opposite-sided command separation required")

    def measure(left, right, denominator):
        if denominator == 0:
            return dict(
                available=False, actual_normalized_delta=0.0, per_normalized_input=None, per_physical_goal_delta=None
            )
        values = {k: v / denominator for k, v in difference(left, right).items()}
        return dict(
            available=True,
            actual_normalized_delta=denominator,
            per_normalized_input=values,
            per_physical_goal_delta={k: v / scale for k, v in values.items()},
        )

    m, p = measure(base, negative, dm), measure(base, positive, dp)
    disagreement = None
    if m["available"] and p["available"]:
        disagreement = {
            k: p["per_normalized_input"][k] - m["per_normalized_input"][k] for k in p["per_normalized_input"]
        }
    return dict(
        axis=axis,
        physical_delta_per_normalized_input=scale,
        minus=m,
        plus=p,
        central=measure(negative, positive, dp - dm),
        one_sided_disagreement=disagreement,
        rotation_convention="world Log(R_right R_left^T); no Euler subtraction",
    )
