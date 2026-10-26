"""239 continuous-goal coupled pose proposal; native decode/commit remains mandatory."""

import numpy as np
from scipy.linalg import block_diag, expm, expm_frechet
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from s2a.current_coupled_response import build_response, rotation_spring
from s2a.geometry import finite, validate_transform
from s2a.osc import OSCInterface


def hat(vector):
    x, y, z = finite(vector, (3,))
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def left_jacobian(vector):
    vector = finite(vector, (3,))
    angle = np.linalg.norm(vector)
    squared = angle * angle
    if angle < 1e-4:
        b, c = 0.5 - squared / 24 + squared**2 / 720, 1 / 6 - squared / 120 + squared**2 / 5040
    else:
        b, c = (1 - np.cos(angle)) / squared, (angle - np.sin(angle)) / angle**3
    skew = hat(vector)
    return np.eye(3) + b * skew + c * skew @ skew


def right_inverse_jacobian(vector):
    vector = finite(vector, (3,))
    angle = np.linalg.norm(vector)
    if angle >= np.pi - 1e-6:
        raise ValueError("SO3 logarithm branch at pi is not differentiable")
    squared = angle * angle
    coefficient = (
        1 / 12 + squared / 720 + squared**2 / 30240
        if angle < 1e-4
        else (1 - 0.5 * angle / np.tan(0.5 * angle)) / squared
    )
    skew = hat(vector)
    return np.eye(3) + 0.5 * skew + coefficient * skew @ skew


class FullCoupledPoseModel:
    """Same228 goal-local spring; Frechet derivatives include its goal dependence."""

    def __init__(self, mass, jacobian, kp, kd):
        self.base = build_response(mass, jacobian, kp, kd, np.eye(3))
        self.coupling = self.base["acceleration_coupling"]
        self.kp, self.kd = self.base["kp"], self.base["kd"]
        self.dt = 0.05

    def response(self, delta, twist, *, derivatives=False, fixed_relative_goal=None):
        delta, twist = finite(delta, (6,)), finite(twist, (6,))
        relative = (
            Rotation.from_rotvec(delta[3:]).as_matrix()
            if fixed_relative_goal is None
            else finite(fixed_relative_goal, (3, 3))
        )
        forcing, stiffness = rotation_spring(relative)
        generator = np.zeros((18, 18))
        generator[:6, 6:12] = np.eye(6)
        generator[6:12, :6] = -self.coupling @ block_diag(np.diag(self.kp[:3]), np.diag(self.kp[3:]) @ stiffness)
        generator[6:12, 6:12] = -self.coupling @ np.diag(self.kd)
        generator[6:12, 12:] = self.coupling @ np.diag(self.kp)
        state = np.r_[np.zeros(6), twist, delta[:3], forcing]
        transition = expm(generator * self.dt)
        prediction = (transition @ state)[:12]
        if not derivatives:
            return prediction
        jacobian = np.zeros((12, 6))
        jacobian[:, :3] = transition[:12, 12:15]
        if fixed_relative_goal is None:
            left = left_jacobian(delta[3:])
            for axis in range(3):
                dr = hat(left[:, axis]) @ relative
                dforcing = 0.5 * np.array([dr[2, 1] - dr[1, 2], dr[0, 2] - dr[2, 0], dr[1, 0] - dr[0, 1]])
                dstiffness = 0.5 * (np.trace(dr) * np.eye(3) - dr.T)
                derivative = np.zeros((18, 18))
                derivative[6:12, 3:6] = -self.coupling[:, 3:] @ np.diag(self.kp[3:]) @ dstiffness
                dstate = np.r_[np.zeros(15), dforcing]
                differential = expm_frechet(generator * self.dt, derivative * self.dt, compute_expm=False)
                jacobian[:, axis + 3] = (differential @ state + transition @ dstate)[:12]
        return prediction, jacobian


def task_objective(
    model, command, interface, measured, desired, twist, reference_twist, innovation, *, fixed_relative_goal=None
):
    """Target-minus-prediction residual, with world SO3 innovation composed on the left."""
    measured, desired = validate_transform(measured), validate_transform(desired)
    if measured.shape != (4, 4) or desired.shape != (4, 4):
        raise ValueError("Individual current and desired SITE poses required")
    command, twist, reference_twist = finite(command, (6,)), finite(twist, (6,)), finite(reference_twist, (6,))
    bias = finite(innovation, (12,))
    if np.any(command < interface.input_min) or np.any(command > interface.input_max):
        raise ValueError("Feasible actual command required")
    delta = interface.decode(command)
    prediction, jacobian = model.response(delta, twist, derivatives=True, fixed_relative_goal=fixed_relative_goal)
    jacobian *= ((interface.output_max - interface.output_min) / (interface.input_max - interface.input_min))[None, :]
    correction = Rotation.from_rotvec(bias[3:6])
    rotation_motion = Rotation.from_rotvec(prediction[3:6])
    predicted_rotation = correction * rotation_motion * Rotation.from_matrix(measured[:3, :3])
    rotation_error = (Rotation.from_matrix(desired[:3, :3]) * predicted_rotation.inv()).as_rotvec()
    predicted_position = measured[:3, 3] + prediction[:3] + bias[:3]
    predicted_twist = prediction[6:] + bias[6:] / model.dt
    rotation_derivative = (
        -right_inverse_jacobian(rotation_error)
        @ correction.as_matrix()
        @ left_jacobian(prediction[3:6])
        @ jacobian[3:6]
    )
    weights = np.r_[
        np.full(3, 1 / 0.01), np.full(3, 1 / 0.05), np.full(3, model.dt / 0.01), np.full(3, model.dt / 0.05)
    ]
    residual = np.r_[desired[:3, 3] - predicted_position, rotation_error, reference_twist - predicted_twist] * weights
    differential = np.vstack([-jacobian[:3], rotation_derivative, -jacobian[6:]]) * weights[:, None]
    pose = np.eye(4)
    pose[:3, 3], pose[:3, :3] = predicted_position, predicted_rotation.as_matrix()
    return dict(
        objective=0.5 * float(residual @ residual),
        gradient=differential.T @ residual,
        residual=residual,
        physical_delta=delta,
        nominal_response=prediction,
        predicted_pose=pose,
        predicted_twist=predicted_twist,
    )


def projected_gradient(gradient, command, lower, upper):
    gradient = np.asarray(gradient, dtype=float).copy()
    gradient[np.isclose(command, lower, atol=1e-9, rtol=0)] = np.minimum(
        gradient[np.isclose(command, lower, atol=1e-9, rtol=0)], 0
    )
    gradient[np.isclose(command, upper, atol=1e-9, rtol=0)] = np.maximum(
        gradient[np.isclose(command, upper, atol=1e-9, rtol=0)], 0
    )
    return gradient


def solve_pose(
    model,
    measured,
    desired,
    twist,
    reference_twist,
    previous_command,
    previous_goal,
    contract,
    *,
    innovation,
    initial_command=None,
):
    """Compare locally stationary reset/latch proposals; no global optimum claim."""
    if contract["control_freq"] != 20 or contract["control_frame"] != "MuJoCo world":
        raise ValueError("Fixed20Hz world controller required")
    for key, actual in (("kp", model.kp), ("kd", model.kd)):
        if not np.allclose(finite(contract[key], (6,)), actual, atol=1e-12, rtol=0):
            raise ValueError("Actual controller gains differ")
    interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    previous = finite(previous_command, (6,))
    if np.any(previous < interface.input_min) or np.any(previous > interface.input_max):
        raise ValueError("Feasible actual previous command required")
    measured = validate_transform(measured)
    if measured.shape != (4, 4):
        raise ValueError("Individual current SITE pose required")
    previous_goal = finite(previous_goal, (3, 3))
    relative_latch = previous_goal @ measured[:3, :3].T
    rotation_spring(relative_latch)
    lower, upper = np.maximum(interface.input_min, previous - 0.25), np.minimum(interface.input_max, previous + 0.25)
    slope = (interface.output_max - interface.output_min) / (interface.input_max - interface.input_min)
    zero = interface.input_min - interface.output_min / slope
    start = previous if initial_command is None else finite(initial_command, (6,))
    start = np.clip(start, lower, upper)
    candidates = []
    modes = [("reset", np.arange(6), None)]
    if np.all(zero[3:] >= lower[3:]) and np.all(zero[3:] <= upper[3:]):
        modes.append(("latch", np.arange(3), relative_latch))
    for mode, axes, fixed in modes:
        template = start.copy()
        if mode == "latch":
            template[3:] = zero[3:]
            if np.any(interface.decode(template)[3:] != 0):
                raise ValueError("Zero physical rotation has no exact normalized representation")

        def evaluate(values):
            command = template.copy()
            command[axes] = values
            result = task_objective(
                model,
                command,
                interface,
                measured,
                desired,
                twist,
                reference_twist,
                innovation,
                fixed_relative_goal=fixed,
            )
            return result["objective"], result["gradient"][axes]

        optimized = minimize(
            evaluate,
            template[axes],
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(lower[axes], upper[axes], strict=True)),
            options=dict(ftol=1e-15, gtol=1e-9, maxiter=200, maxls=40),
        )
        command = template.copy()
        command[axes] = np.clip(optimized.x, lower[axes], upper[axes])
        adjusted = False
        if mode == "reset" and np.all(interface.decode(command)[3:] == 0):
            # A nonzero, explicit physical representative triggers native reset.
            # It is checked after adjustment; it is not silently treated as latch.
            for axis in range(3, 6):
                for sign in (1, -1):
                    value = zero[axis] + sign * 1e-10 / slope[axis]
                    if lower[axis] <= value <= upper[axis]:
                        command[axis], adjusted = value, True
                        break
                if adjusted:
                    break
            if not adjusted or np.all(interface.decode(command)[3:] == 0):
                raise ValueError("No feasible nonzero native reset representative")
        result = task_objective(
            model, command, interface, measured, desired, twist, reference_twist, innovation, fixed_relative_goal=fixed
        )
        pg = projected_gradient(result["gradient"][axes], command[axes], lower[axes], upper[axes])
        stationarity = float(np.max(np.abs(pg)))
        if stationarity > 2e-6:
            raise RuntimeError(f"{mode} proposal not locally stationary: {stationarity}")
        result.update(
            command=command,
            goal_mode=mode,
            lower=lower.copy(),
            upper=upper.copy(),
            innovation=np.asarray(innovation).copy(),
            solver_iterations=int(optimized.nit),
            optimizer_success=bool(optimized.success),
            optimizer_message=str(optimized.message),
            max_projected_gradient=stationarity,
            reset_representative_adjusted=adjusted,
            fixed_relative_goal=None if fixed is None else fixed.copy(),
            global_optimality_claim=False,
            native_decode_and_actual_commit_required=True,
        )
        candidates.append(result)
    chosen = min(candidates, key=lambda result: (result["objective"], result["goal_mode"] != "latch")).copy()
    chosen["mode_objectives"] = {result["goal_mode"]: result["objective"] for result in candidates}
    chosen["mode_candidates"] = candidates
    return chosen
