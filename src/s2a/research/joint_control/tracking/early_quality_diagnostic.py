"""Past-only model-error envelopes and independent quality-witness checks."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, pose_error
from s2a.research.joint_control.limits.joint_preview_constraint_audit import audit_joint_preview_constraints
from s2a.research.response_models.causal.causal_inner_error import CausalInnerError
from s2a.research.response_models.forward.joint_limit_response import build_joint_response, predict_joint_response


def state_before_period(initial_state, initial_command, periods, start_step, offset):
    """Return only current state, prior action and the already-generated plan."""
    if type(start_step) is not int or type(offset) is not int or not 0 <= offset < len(periods):
        raise ValueError("A bounded current period is required")
    if periods[offset]["step"] != start_step + offset:
        raise ValueError("Current period identity mismatch")
    if offset:
        preceding = periods[offset - 1]
        if preceding["step"] != start_step + offset - 1:
            raise ValueError("Previous completed period identity mismatch")
        state, previous = preceding["actual_state"], preceding["command"]
    else:
        state, previous = initial_state, initial_command
    plan = finite(periods[offset]["decision"]["refined_plan"], (8, 6)).copy()
    return finite(state, (79,)).copy(), finite(previous, (7,)).copy(), plan


def completed_error_profile(periods, start_step, offset, forecast):
    """Stage each prediction before observing that completed period's q samples."""
    if type(start_step) is not int or type(offset) is not int or not 0 <= offset <= len(periods):
        raise ValueError("Bounded completed-period prefix required")
    history = CausalInnerError()
    for index in range(offset):
        period = periods[index]
        if period["step"] != start_step + index:
            raise ValueError("Consecutive completed periods required")
        prediction = finite(forecast(index), (25, 7))
        history.stage(period["step"], prediction)
        actual = finite([item["q"] for item in period["actual_inner"]], (25, 7))
        history.complete(period["step"], actual)
    return history.snapshot()


def independently_check_witness(
    witness,
    mass,
    jacobian,
    kp,
    kd,
    q,
    velocity,
    initial_joint,
    interface,
    previous,
    site,
    site_from_tcp,
    first_target,
    lower,
    upper,
    *,
    past_inner_error_rad,
    enforce_pose_caps=True,
    enforce_braking=True,
):
    """Rebuild each period from the scalar transition, then audit constraints."""
    if type(enforce_pose_caps) is not bool:
        raise ValueError("Explicit boolean first-pose enforcement required")
    if type(enforce_braking) is not bool:
        raise ValueError("Explicit boolean extra-braking enforcement required")
    if witness.get("witness_found") is not True:
        raise ValueError("A claimed model witness is required")
    commands = finite(witness["planned_commands"])
    if commands.ndim != 2 or commands.shape[1] != 6 or not 1 <= len(commands) <= 8:
        raise ValueError("Bounded six-dimensional plan required")
    q = finite(q, (7,)).copy()
    initial_q = q.copy()
    velocity = finite(velocity, (7,)).copy()
    inner, endpoints, velocities = [], [], []
    for command in commands:
        model = build_joint_response(mass, jacobian, kp, kd, q, initial_joint, null_kp=10.0, dt=0.002)
        delta = interface.decode(command)
        for substeps in range(1, 26):
            result = predict_joint_response(
                model, velocity, delta, lower, upper, substeps=substeps, enforce_limits=False
            )
            inner.append(result["joint_position"].copy())
        q, velocity = result["joint_position"].copy(), result["joint_velocity"].copy()
        endpoints.append(q.copy())
        velocities.append(velocity.copy())
    independent = {
        "inner_joint_position": np.asarray(inner),
        "joint_position": np.asarray(endpoints),
        "joint_velocity": np.asarray(velocities),
    }
    differences = {
        key: float(np.max(np.abs(value - np.asarray(witness["rollout"][key])))) for key, value in independent.items()
    }
    if any(value > 1e-10 for value in differences.values()):
        raise ValueError("Independent joint response differs from the witness rollout")
    audit = audit_joint_preview_constraints(
        independent,
        commands,
        previous,
        interface,
        lower,
        upper,
        past_inner_error_rad=past_inner_error_rad,
        terminal_normals=witness["terminal_normals"],
        braking_time_s=0.1,
    )
    brake_names = ("braking_lower", "braking_upper")
    brake_ok = all(not len(audit["groups"][name]["violated_indices"]) for name in brake_names)
    included_ok = all(
        not len(group["violated_indices"])
        for name, group in audit["groups"].items()
        if enforce_braking or name not in brake_names
    )
    if not included_ok:
        raise ValueError("Independent joint, rate, braking or terminal audit failed")
    movement = np.asarray(jacobian) @ (independent["joint_position"][0] - initial_q)
    predicted_site = finite(site, (4, 4)).copy()
    predicted_site[:3, 3] += movement[:3]
    predicted_site[:3, :3] = Rotation.from_rotvec(movement[3:]).as_matrix() @ predicted_site[:3, :3]
    error = pose_error(first_target, predicted_site @ site_from_tcp)
    np.testing.assert_allclose(error, witness["first_tcp_error"], atol=1e-10, rtol=0)
    pose_ok = bool(np.linalg.norm(error[:3]) <= 0.01 + 1e-10 and np.linalg.norm(error[3:]) <= 0.05 + 1e-10)
    if enforce_pose_caps and not pose_ok:
        raise ValueError("Independent absolute first-pose caps failed")
    return {
        "joint_rollout_max_difference": differences,
        "constraint_minima": {name: group["minimum"] for name, group in audit["groups"].items()},
        "first_tcp_error": error,
        "independent_period_model_builds": len(commands),
        "independent_endpoint_queries": 25 * len(commands),
        "independent_transition_steps": 325 * len(commands),
        "first_pose_caps_satisfied": pose_ok,
        "first_pose_caps_enforced": enforce_pose_caps,
        "braking_constraint_satisfied": brake_ok,
        "braking_constraint_enforced": enforce_braking,
        "passed": True,
    }
