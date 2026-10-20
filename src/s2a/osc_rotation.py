"""Bounded inverse of the fixed OSC orientation spring.

The robosuite delta interface sets ``R_goal = Exp(delta_world) @ R_now``.
Its orientation spring uses ``sin(theta) * axis``, not the SO(3) logarithm.
The model here integrates that held goal with measured world angular velocity
and fixed rotational kp / kd. It does not model contact, joint limits, torque
saturation, nullspace motion, or imperfect operational-space decoupling.

This module is independent of the simulator and has no task or object inputs.
The original OSC converter remains available as a separate comparison.
"""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from s2a.geometry import finite

ROTATION_MATRIX_ATOL = 1e-6


def _rotation(value):
    value = finite(value, (3, 3))
    # Upstream robosuite quat2mat explicitly casts its quaternion to float32.
    # Preserve that raw goal for the spring rather than silently projecting
    # it to a different SO(3) matrix. The accepted numerical defect is reported.
    if not np.allclose(value.T @ value, np.eye(3), atol=ROTATION_MATRIX_ATOL, rtol=0) or not np.isclose(
        np.linalg.det(value), 1, atol=ROTATION_MATRIX_ATOL, rtol=0
    ):
        raise ValueError("orientation must be a proper rotation matrix")
    return value


def _gains(kp, kd, dt, integration_steps):
    kp, kd = finite(kp, (3,)), finite(kd, (3,))
    if np.any(kp <= 0) or np.any(kd <= 0):
        raise ValueError("fixed rotational gains must be positive")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("planned dt must be positive")
    if isinstance(integration_steps, bool) or not isinstance(integration_steps, (int, np.integer)):
        raise ValueError("integration_steps must be a positive integer")
    if integration_steps < 1:
        raise ValueError("integration_steps must be a positive integer")
    return kp, kd


def _spring(goal, current):
    # Same world-frame spring as robosuite's column cross products. This
    # expression also avoids the log's branch cut for the internal dynamics.
    relative = goal @ current.T
    return 0.5 * np.array(
        [relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]]
    )


def _integrate_relative(delta, omega, kp, kd, dt, steps, *, retained_goal=None):
    """Second-order Lie midpoint integration; no quaternion renormalization."""
    goal = Rotation.from_rotvec(delta).as_matrix() if retained_goal is None else retained_goal
    current, omega = np.eye(3), omega.copy()
    interval = dt / steps
    for _ in range(steps):
        acceleration = kp * _spring(goal, current) - kd * omega
        midpoint_omega = omega + 0.5 * interval * acceleration
        midpoint_rotation = Rotation.from_rotvec(0.5 * interval * omega).as_matrix() @ current
        midpoint_acceleration = kp * _spring(goal, midpoint_rotation) - kd * midpoint_omega
        current = Rotation.from_rotvec(interval * midpoint_omega).as_matrix() @ current
        omega += interval * midpoint_acceleration
    return current, omega


def predict_osc_orientation(
    measured_rotation,
    physical_goal_delta,
    world_angular_velocity,
    *,
    kp,
    kd,
    dt,
    integration_steps=50,
    current_goal_rotation=None,
):
    """Predict one held-goal interval under the local rotational OSC model.

    Rotations map control-point coordinates into world; angular velocity and
    the physical goal's rotation vector are world-frame quantities in rad/s
    and rad. Returned orientation is absolute, and velocity is world-frame.
    All absolute rotations must refer to the SAME frame, including the current
    goal. LIBERO's observed eef BODY rotation differs from its OSC SITE frame;
    the caller must apply their fixed right-offset before mixing those inputs.
    A zero physical rotation delta preserves the controller's previous goal,
    so that case requires ``current_goal_rotation``. This prediction is a
    model diagnostic, not a simulated robot observation.
    """
    measured = _rotation(measured_rotation)
    delta, omega = finite(physical_goal_delta, (3,)), finite(world_angular_velocity, (3,))
    kp, kd = _gains(kp, kd, dt, integration_steps)
    retained = None
    if np.all(delta == 0):
        if current_goal_rotation is None:
            raise ValueError("zero rotational command requires current_goal_rotation")
        retained = _rotation(current_goal_rotation) @ measured.T
    relative, final_omega = _integrate_relative(delta, omega, kp, kd, dt, integration_steps, retained_goal=retained)
    return relative @ measured, final_omega


def bounded_osc_rotation_inverse(
    measured_rotation,
    desired_rotation,
    world_angular_velocity,
    *,
    interface,
    kp,
    kd,
    dt,
    previous_command=None,
    current_goal_rotation=None,
    max_command_rate=None,
    integration_steps=50,
    max_nfev=30,
    zero_goal_epsilon_rad=None,
):
    """Return a legal three-component rotational OSC command for one endpoint.

    All three absolute rotations, including current_goal_rotation, must map the
    SAME controlled frame to world. A caller using LIBERO's eef BODY observation
    must convert it and the desired orientation to the controller SITE frame,
    or convert the retained controller goal into the body frame. A world-frame
    angular velocity alone does not remove this absolute-frame requirement.

    ``interface`` is an OSCInterface with the actual six-dimensional affine
    input/output limits. ``previous_command`` is only its three normalized
    rotational entries. When a rate limit is supplied, that previous command
    is required; the optimization box intersects the affine and rate limits
    *before* solving. The caller owns history and resets it between trials.

    The requested endpoint can be unreachable in dt. In that case this solves
    for a bounded local-model approximation and reports the residual; it does
    not claim that the physical robot will achieve it. Bounds must keep every
    possible goal vector inside pi/2, where the spring's radial gain remains
    monotone. The ordinary LIBERO +/-0.5 rad per component satisfies this.

    A strictly zero *physical* rot command causes historical robosuite to
    retain its old orientation goal. Pass ``current_goal_rotation`` from the
    current controller state to compare that discrete hold candidate with the
    nonzero fresh-goal candidates. Without that state, a zero candidate cannot
    be predicted. ``zero_goal_epsilon_rad`` explicitly opts into comparing an
    additional tiny, nonzero fresh goal when the continuous inverse gives zero;
    it is disabled by default and its use is always reported.
    """
    measured, desired = _rotation(measured_rotation), _rotation(desired_rotation)
    omega = finite(world_angular_velocity, (3,))
    retained_goal = None if current_goal_rotation is None else _rotation(current_goal_rotation) @ measured.T
    rotation_inputs = dict(measured_rotation=measured, desired_rotation=desired)
    if current_goal_rotation is not None:
        rotation_inputs["current_goal_rotation"] = np.asarray(current_goal_rotation, dtype=float)
    input_rotation_defects = {
        name: dict(
            orthogonality_max_abs=float(np.max(np.abs(value.T @ value - np.eye(3)))),
            determinant_error_abs=float(abs(np.linalg.det(value) - 1)),
        )
        for name, value in rotation_inputs.items()
    }
    kp, kd = _gains(kp, kd, dt, integration_steps)
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    if zero_goal_epsilon_rad is not None and (
        not np.isfinite(zero_goal_epsilon_rad) or not 0 < zero_goal_epsilon_rad <= 1e-6
    ):
        raise ValueError("zero-goal reanchor epsilon must be positive and at most 1e-6 rad")
    input_min, input_max = interface.input_min[3:], interface.input_max[3:]
    output_min, output_max = interface.output_min[3:], interface.output_max[3:]
    if np.linalg.norm(np.maximum(np.abs(output_min), np.abs(output_max))) >= np.pi / 2:
        raise ValueError("rotation goal box must stay within the monotone pi/2 spring region")
    lower, upper = input_min.copy(), input_max.copy()
    if max_command_rate is not None:
        if not np.isfinite(max_command_rate) or max_command_rate <= 0:
            raise ValueError("max_command_rate must be positive")
        if previous_command is None:
            raise ValueError("rate limit requires previous rotational command")
        previous = finite(previous_command, (3,))
        if np.any(previous < input_min) or np.any(previous > input_max):
            raise ValueError("previous command is outside the affine interface")
        lower = np.maximum(lower, previous - max_command_rate * dt)
        upper = np.minimum(upper, previous + max_command_rate * dt)

    # Mirror robosuite's centered affine operation order. In an offset
    # interface normalized command zero need not mean physical delta zero.
    scale = (output_max - output_min) / (input_max - input_min)
    input_midpoint, output_midpoint = (input_min + input_max) / 2, (output_min + output_max) / 2

    def decode(command):
        return (command - input_midpoint) * scale + output_midpoint

    def encode(delta):
        return input_midpoint + (delta - output_midpoint) / scale

    desired_relative = desired @ measured.T
    endpoint_error = Rotation.from_matrix(desired_relative).as_rotvec()
    response, momentum = np.empty(3), np.empty(3)
    for i in range(3):
        evolution = expm(np.array([[0.0, 1.0], [-kp[i], -kd[i]]]) * dt)
        response[i], momentum[i] = 1 - evolution[0, 0], evolution[0, 1]
    if np.any(response < 1e-5):
        raise ValueError("planned interval has insufficient inverse response")

    def residual(command):
        predicted, _ = _integrate_relative(decode(command), omega, kp, kd, dt, integration_steps)
        return Rotation.from_matrix(desired_relative @ predicted.T).as_rotvec()

    seeds = [
        np.clip(encode(endpoint_error), lower, upper),
        np.clip(encode((endpoint_error - momentum * omega) / response), lower, upper),
    ]
    initial = min(seeds, key=lambda value: np.linalg.norm(residual(value)))
    initial_error = float(np.linalg.norm(residual(initial)))
    solved = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=max_nfev,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    candidates = []

    def add_candidate(value, name, *, epsilon_used=False):
        value = np.clip(value, lower, upper)
        delta = decode(value)
        preserve = bool(np.all(delta == 0))
        if preserve and retained_goal is None:
            return
        predicted, final_omega = _integrate_relative(
            delta,
            omega,
            kp,
            kd,
            dt,
            integration_steps,
            retained_goal=retained_goal if preserve else None,
        )
        norm = float(np.linalg.norm(Rotation.from_matrix(desired_relative @ predicted.T).as_rotvec()))
        candidates.append((norm, value, delta, predicted, final_omega, preserve, epsilon_used, name))

    for value, name in [(seeds[0], "baseline_seed"), (seeds[1], "linear_inverse_seed"), (solved.x, "solver")]:
        add_candidate(value, name)
        if np.all(decode(value) == 0) and zero_goal_epsilon_rad is not None:
            physical_lower, physical_upper = decode(lower), decode(upper)
            for axis in range(3):
                delta = np.zeros(3)
                delta[axis] = min(zero_goal_epsilon_rad, physical_upper[axis])
                if delta[axis] <= 0:
                    delta[axis] = max(-zero_goal_epsilon_rad, physical_lower[axis])
                epsilon_command = np.clip(encode(delta), lower, upper)
                if np.any(decode(epsilon_command) != 0):
                    add_candidate(epsilon_command, "explicit_nonzero_reanchor", epsilon_used=True)
                    break
    neutral = encode(np.zeros(3))
    zero_candidate_available = bool(
        np.all(neutral >= lower) and np.all(neutral <= upper) and np.all(decode(neutral) == 0)
    )
    if zero_candidate_available:
        add_candidate(neutral, "preserve_current_goal")
    if not candidates:
        raise ValueError("zero rotational command requires current_goal_rotation or explicit nonzero reanchor")
    _, command, sent_delta, predicted, final_omega, preserved, epsilon_used, chosen = min(
        candidates, key=lambda item: item[0]
    )
    predicted_error = Rotation.from_matrix(desired_relative @ predicted.T).as_rotvec()
    bound_tolerance = 1e-7
    interface_bound_active = bool(
        np.any(np.isclose(command, input_min, atol=bound_tolerance, rtol=0))
        or np.any(np.isclose(command, input_max, atol=bound_tolerance, rtol=0))
    )
    rate_bound_active = bool(
        max_command_rate is not None
        and (
            np.any((lower > input_min) & np.isclose(command, lower, atol=bound_tolerance, rtol=0))
            or np.any((upper < input_max) & np.isclose(command, upper, atol=bound_tolerance, rtol=0))
        )
    )
    return dict(
        command=command.copy(),
        sent_delta=sent_delta.copy(),
        normalized_lower_bound=lower,
        normalized_upper_bound=upper,
        physical_lower_bound=decode(lower),
        physical_upper_bound=decode(upper),
        predicted_rotation=predicted @ measured,
        predicted_world_angular_velocity=final_omega,
        endpoint_error_rad=float(np.linalg.norm(endpoint_error)),
        predicted_endpoint_error_rad=float(np.linalg.norm(predicted_error)),
        continuous_initial_seed_error_rad=initial_error,
        best_actual_candidate_error_rad=min(item[0] for item in candidates),
        solver_converged=bool(solved.success),
        solver_status=int(solved.status),
        solver_nfev=int(solved.nfev),
        chosen_candidate=chosen,
        used_solver_result=chosen == "solver",
        interface_bound_active=interface_bound_active,
        rate_bound_active=rate_bound_active,
        zero_command_candidate_available=zero_candidate_available and retained_goal is not None,
        zero_command_preserved_goal=preserved,
        zero_goal_epsilon_used=epsilon_used,
        current_goal_rotation_provided=current_goal_rotation is not None,
        applied_goal_rotation=(retained_goal if preserved else Rotation.from_rotvec(sent_delta).as_matrix()) @ measured,
        command_dtype=str(command.dtype),
        input_rotation_defects=input_rotation_defects,
        rotation_matrix_tolerance=ROTATION_MATRIX_ATOL,
        input_rotations_projected=False,
        integration_steps=int(integration_steps),
        local_model_only=True,
    )
