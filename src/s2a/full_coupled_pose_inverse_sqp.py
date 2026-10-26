"""244 shared239 physics and a universally applied SQP stationarity solver."""

import numpy as np
from scipy.optimize import minimize

from s2a.current_coupled_response import rotation_spring
from s2a.full_coupled_pose_inverse import FullCoupledPoseModel, projected_gradient, task_objective
from s2a.geometry import finite, validate_transform
from s2a.osc import OSCInterface

__all__ = ["FullCoupledPoseModel", "solve_pose"]


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
            method="SLSQP",
            bounds=list(zip(lower[axes], upper[axes], strict=True)),
            options=dict(ftol=1e-13, maxiter=500),
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
            solver_method="SLSQP",
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
