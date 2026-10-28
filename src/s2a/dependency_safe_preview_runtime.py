"""Generated-chunk preview plus explicit continuous first-command forecast for native commit."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.dependency_safe_preview_executable import plan_pose
from s2a.full_coupled_pose_inverse import FullCoupledPoseModel
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import finite, validate_transform
from s2a.osc import OSCInterface


def solve_preview(
    mass,
    jacobian,
    measured,
    twist,
    generated_poses,
    query_anchor,
    current_index,
    previous_command,
    contract,
    *,
    innovation,
):
    """References come from one already generated chunk; future observations are not inputs.

    The tangent preview prediction and goal-dependent first-command forecast are
    deliberately separate. The caller must decode/commit the actual native goal.
    """
    generated_poses, query_anchor = validate_transform(generated_poses), validate_transform(query_anchor)
    measured = validate_transform(measured)
    if (
        generated_poses.ndim != 3
        or query_anchor.shape != (4, 4)
        or measured.shape != (4, 4)
        or type(current_index) is not int
        or not 0 <= current_index < len(generated_poses)
    ):
        raise ValueError("Current index into one generated chunk and individual poses required")
    end = min(current_index + 4, len(generated_poses))
    references = generated_poses[current_index:end].copy()
    reference_twists = np.asarray(
        [reference_world_twist(generated_poses, i, query_anchor, 0.05)["twist"] for i in range(current_index, end)]
    )
    plan = plan_pose(
        measured, twist, references, reference_twists, previous_command, mass, jacobian, contract, innovation=innovation
    )
    command = plan["command"].copy()
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    delta = interface.decode(command)
    if np.all(delta[3:] == 0):
        raise ValueError("Executable preview must trigger native reset")
    model = FullCoupledPoseModel(mass, jacobian, contract["kp"], contract["kd"])
    response = model.response(delta, twist)
    bias = finite(innovation, (12,))
    predicted = measured.copy()
    predicted[:3, 3] += response[:3] + bias[:3]
    predicted[:3, :3] = (
        Rotation.from_rotvec(bias[3:6]) * Rotation.from_rotvec(response[3:6]) * Rotation.from_matrix(measured[:3, :3])
    ).as_matrix()
    return dict(
        command=command,
        goal_mode="reset",
        physical_delta=delta,
        nominal_response=response,
        predicted_pose=predicted,
        predicted_twist=response[6:] + bias[6:] / 0.05,
        preview=plan,
        references=references,
        reference_twists=reference_twists,
        reference_current_index=current_index,
        reference_end_index=end,
        continuous_forecast_is_preview_prediction=False,
        native_decode_and_actual_commit_required=True,
        nonlinear_optimality_claim=False,
    )
