"""A bounded causal proposal to defer a closing motor command before release."""

import numpy as np

from s2a.research.gripper.coupled_gripper_preview import coupled_opening_preview
from s2a.research.gripper.gripper_tracking import panda_opening_from_motor_target


def propose_deferred_close(poses, openings, index, measured_tcp, measured_opening, motor_target, baseline, closing):
    """Propose retaining the motor target under the fixed release-evidence rule.

    The caller supplies only the original current decision, measured state and
    already-generated reference. The proposal does not advance policy history
    or certify contact retention; execution requires its own transaction.
    """
    motor = np.asarray(motor_target, dtype=float)
    if (
        isinstance(measured_opening, (bool, np.bool_))
        or not np.isscalar(measured_opening)
        or not np.isfinite(measured_opening)
        or not 0 <= measured_opening <= 1
        or isinstance(baseline, (bool, np.bool_))
        or not np.isscalar(baseline)
        or not np.isfinite(baseline)
        or not -1 <= baseline <= 1
        or type(closing) is not bool
        or motor.shape != (2,)
        or not np.isfinite(motor).all()
        or np.any(abs(motor) > 1)
        or abs(float(motor.sum())) > 1e-10
    ):
        raise ValueError("finite measured state, antisymmetric motor and original decision required")
    preview = coupled_opening_preview(
        poses,
        openings,
        index,
        measured_tcp,
        preview_steps=6,
        position_span_max_m=0.01,
        rotation_span_max_rad=0.1,
        measured_position_max_m=0.01,
        measured_rotation_max_rad=0.05,
    )
    proposal = preview["preview_proposal"]
    guards = preview["guards"]
    motor_opening = panda_opening_from_motor_target(motor)
    rise = proposal["reference"] - proposal["original_reference"]
    defer = bool(
        baseline == 1.0
        and closing
        and motor_opening <= measured_opening
        and index + 6 < len(openings)
        and proposal["advanced"]
        and rise >= 0.01 / 0.079
        and not guards["position_span"]
        and guards["rotation_span"]
        and guards["measured_position"]
        and guards["measured_rotation"]
    )
    return dict(
        command=0.0 if defer else float(baseline),
        deferred=defer,
        motor_opening=motor_opening,
        predicted_rise=float(rise),
        preview=preview,
    )
