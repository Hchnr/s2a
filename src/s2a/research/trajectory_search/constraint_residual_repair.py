"""One bounded correction of observed constraint residuals using saved FD data.

This module never calls a trajectory predictor. QP qualification and numerical
repair eligibility do not confer physical control qualification.
"""

from copy import deepcopy

import numpy as np

from s2a.accepted_point_sqp_core import solve_epigraph_qp
from s2a.geometry import finite, pose_error, validate_transform
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints
from s2a.tail_translation_reserve import SAFETY_NAMES

RADIUS = 0.0025
STAGE_RADIUS = 0.005
SAFETY_BUFFER = 1e-8
PREFIX_MARGIN = 1e-4
ALPHAS = tuple(2.0**-index for index in range(9))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _squared(entry):
    errors = finite(entry["tcp_errors"], (12, 6))
    squared = finite(
        np.column_stack((np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2))
    )
    if "normalized_squared_errors" in entry:
        _require(
            np.array_equal(squared, finite(entry["normalized_squared_errors"], (12, 2))),
            "Saved normalized errors must exactly match the original pose-cap algebra",
        )
    return squared


def build_repair_problem(base, trial, probes, nominal_deltas, previous, input_min, input_max, stage_lower, stage_upper):
    """Use all 72 saved signed FD probes and the trial's observed intercepts."""
    x, y = finite(base["commands"], (12, 6)).copy(), finite(trial["commands"], (12, 6)).copy()
    previous = finite(previous, (6,)).copy()
    input_min, input_max = finite(input_min, (6,)).copy(), finite(input_max, (6,)).copy()
    _require(np.all(input_min < input_max), "Strict command bounds")
    stage_lower, stage_upper = finite(stage_lower, (12, 6)).copy(), finite(stage_upper, (12, 6)).copy()
    global_lower, global_upper = np.tile(input_min, (12, 1)), np.tile(input_max, (12, 1))
    global_lower[0] = np.maximum(global_lower[0], previous - 0.25)
    global_upper[0] = np.minimum(global_upper[0], previous + 0.25)
    _require(
        np.array_equal(stage_lower, np.maximum(global_lower, x - STAGE_RADIUS))
        and np.array_equal(stage_upper, np.minimum(global_upper, x + STAGE_RADIUS)),
        "Saved stage bounds must be the original radius0.005 intersected with command bounds",
    )
    for commands in (x, y):
        changes = np.diff(np.vstack((previous, commands)), axis=0)
        _require(
            np.all(commands >= stage_lower - 1e-12)
            and np.all(commands <= stage_upper + 1e-12)
            and np.all(np.abs(changes) <= 0.25 + 1e-12),
            "Base and trial must retain original stage and command admission",
        )
    base_safety, trial_safety = finite(base["safety_slack"], (4369,)), finite(trial["safety_slack"], (4369,))
    base_q, trial_q = _squared(base), _squared(trial)
    _require(np.all(base_safety >= SAFETY_BUFFER) and np.all(base_q[:9] <= 1), "Original feasible-prefix anchor")
    base_target, trial_target = float(np.max(base_q[9])), float(np.max(trial_q[9]))
    _require(trial_target < base_target, "The fixed trial must actually improve cursor9")
    threshold = base_target - 0.5 * (base_target - trial_target)
    deltas = finite(nominal_deltas, (72,)).copy()
    _require(len(probes) == 72 and np.all(np.abs(deltas) == 1e-4), "Complete original signed1e-4 stencil")
    safety_columns, pose_columns, actual_deltas = [], [], []
    for axis, (probe, nominal) in enumerate(zip(probes, deltas, strict=True)):
        command = finite(probe["commands"], (12, 6))
        actual = command.ravel() - x.ravel()
        other_axes = np.arange(72) != axis
        tolerance = 4 * np.finfo(float).eps * max(1.0, abs(x.ravel()[axis]), abs(command.ravel()[axis]))
        _require(
            np.all(actual[other_axes] == 0) and actual[axis] != 0 and abs(actual[axis] - nominal) <= tolerance,
            "Each saved probe must change only its registered axis by its signed nominal displacement",
        )
        _require(
            np.all(command >= stage_lower - 1e-12) and np.all(command <= stage_upper + 1e-12),
            "Original FD probes must remain inside their saved stage bounds",
        )
        safety_columns.append((finite(probe["safety_slack"], (4369,)) - base_safety) / nominal)
        pose_columns.append((_squared(probe).ravel() - base_q.ravel()) / nominal)
        actual_deltas.append(float(actual[axis]))
    Js, Jq = finite(np.asarray(safety_columns).T), finite(np.asarray(pose_columns).T)
    command_lower = np.maximum.reduce((stage_lower, np.tile(input_min, (12, 1)), y - RADIUS))
    command_upper = np.minimum.reduce((stage_upper, np.tile(input_max, (12, 1)), y + RADIUS))
    _require(np.all(command_lower <= command_upper), "Nonempty correction bounds")
    identity = np.eye(72)
    difference, center = identity - np.eye(72, k=-6), np.r_[previous, np.zeros(66)]
    blocks = (
        ("safety", Js, trial_safety - SAFETY_BUFFER),
        ("prefix", -Jq[:18], 1 - PREFIX_MARGIN - trial_q[:9].ravel()),
        ("target", -Jq[18:20], threshold - trial_q[9]),
        ("rate_lower", difference, 0.25 + difference @ y.ravel() - center),
        ("rate_upper", -difference, 0.25 - difference @ y.ravel() + center),
        ("command_lower", identity, (y - command_lower).ravel()),
        ("command_upper", -identity, (command_upper - y).ravel()),
    )
    matrices, offsets, row_blocks, start = [], [], {}, 0
    for name, matrix, offset in blocks:
        matrices.append(np.c_[matrix, np.zeros(len(offset))])
        offsets.append(offset)
        row_blocks[name] = [start, start + len(offset)]
        start += len(offset)
    matrices.append(np.r_[np.zeros(72), 1.0][None])
    offsets.append(np.zeros(1))
    row_blocks["epigraph"] = [start, start + 1]
    return dict(
        base_commands=x,
        trial_commands=y,
        previous=previous,
        input_min=input_min,
        input_max=input_max,
        stage_lower=stage_lower,
        stage_upper=stage_upper,
        command_lower=command_lower,
        command_upper=command_upper,
        delta_lower=command_lower - y,
        delta_upper=command_upper - y,
        radius=RADIUS,
        safety_buffer=SAFETY_BUFFER,
        prefix_optimization_margin=PREFIX_MARGIN,
        base_target=base_target,
        trial_target=trial_target,
        threshold=threshold,
        base_safety=base_safety.copy(),
        trial_safety=trial_safety.copy(),
        base_normalized_squared_errors=base_q.copy(),
        trial_normalized_squared_errors=trial_q.copy(),
        nominal_deltas=deltas,
        actual_deltas=np.asarray(actual_deltas),
        Js=Js,
        Jq=Jq,
        A=finite(np.vstack(matrices)),
        b=finite(np.concatenate(offsets)),
        B=identity,
        g=np.zeros(72),
        initial_t=0.0,
        row_blocks=row_blocks,
        solver_iteration_limit=100,
        certificate_tolerance=1e-6,
        new_response_evaluations=0,
        empirical_error_bound_used=False,
    )


def solve_repair(problem):
    """Make exactly one algebraic QP call; leave scientific failure reportable."""
    return solve_epigraph_qp(
        problem["B"],
        problem["g"],
        problem["A"],
        problem["b"],
        problem["initial_t"],
        max_iterations=100,
        certificate_tolerance=1e-6,
    )


def fixed_repair_commands(problem, qp):
    """Preflight the complete fixed ray before any new response can be called."""
    result = dict(admitted=False, reason="qp_unqualified", commands=[], planned_request_count=9)
    if not qp["qualified"]:
        return result
    point = finite(qp["point"], (73,))
    y = finite(problem["trial_commands"], (12, 6))
    correction = point[:72].reshape(12, 6)
    rows, seen = [], {y.tobytes()}
    for index, alpha in enumerate(ALPHAS):
        command = finite(y + alpha * correction)
        actual = command - y
        changes = np.diff(np.vstack((problem["previous"], command)), axis=0)
        gates = dict(
            box=bool(
                np.all(command >= problem["input_min"] - 1e-12) and np.all(command <= problem["input_max"] + 1e-12)
            ),
            rate=bool(np.all(np.abs(changes) <= 0.25 + 1e-12)),
            stage=bool(
                np.all(command >= problem["stage_lower"] - 1e-12) and np.all(command <= problem["stage_upper"] + 1e-12)
            ),
            radius=bool(np.max(np.abs(actual)) <= RADIUS + 1e-12),
            nonzero=bool(np.any(actual != 0)),
            unique=command.tobytes() not in seen,
        )
        rows.append(dict(alpha_index=index, alpha=alpha, commands=command, actual_delta=actual, admission=gates))
        seen.add(command.tobytes())
    result.update(commands=rows, admitted=all(all(row["admission"].values()) for row in rows))
    result["reason"] = "admitted" if result["admitted"] else "fixed_ray_preflight_failed"
    return result


def evaluate_repair_entry(rollout, commands, previous, interface, lower, upper, desired, tool, profile, normals):
    """Recompute the original full-H12 audit and pose caps from one response."""
    commands, previous = finite(commands, (12, 6)).copy(), finite(previous, (6,)).copy()
    rollout = deepcopy(rollout)
    audit = audit_joint_preview_constraints(
        rollout,
        commands,
        previous,
        interface,
        lower,
        upper,
        past_inner_error_rad=profile,
        terminal_normals=normals,
        braking_time_s=0.1,
    )
    desired, tool, poses = validate_transform(desired), validate_transform(tool), validate_transform(rollout["pose"])
    _require(desired.shape == poses.shape == (12, 4, 4) and tool.shape == (4, 4), "Complete H12 original pose contract")
    errors = pose_error(desired, poses @ tool)
    squared = _squared(dict(tcp_errors=errors))
    failures = np.flatnonzero(np.any(squared > 1, axis=1))
    prefix = int(failures[0]) if len(failures) else 12
    return dict(
        commands=commands,
        rollout=rollout,
        audit=audit,
        tcp_errors=errors,
        normalized_squared_errors=squared,
        safety_slack=np.concatenate([audit["groups"][key]["slack"].ravel() for key in SAFETY_NAMES]),
        prefix_length=prefix,
        maximum_normalized_squared_error=float(np.max(squared)),
        maximum_pose_cap_violation=float(max(0.0, np.max(np.sqrt(squared)) - 1.0)),
        pose_loss=float(np.sum(squared)),
        feasible=bool(audit["feasible"] and prefix == 12),
    )


def classify_repair_entry(entry, threshold):
    """Keep the stricter optimization prefix margin separate from admission."""
    squared = _squared(entry)
    safety = finite(entry["safety_slack"], (4369,))
    threshold = float(finite(threshold, ()))
    gates = dict(
        joint_feasible=bool(entry["audit"]["feasible"]),
        safety_buffer_satisfied=bool(np.all(safety >= SAFETY_BUFFER)),
        prefix9_preserved=bool(np.all(squared[:9] <= 1)),
        descent_preserved=bool(np.all(squared[9] <= threshold)),
    )
    return dict(
        **gates,
        prefix_optimization_margin_satisfied=bool(np.all(squared[:9] <= 1 - PREFIX_MARGIN)),
        repair_eligible=all(gates.values()),
        feasible=bool(entry["audit"]["feasible"] and np.all(squared <= 1)),
    )
