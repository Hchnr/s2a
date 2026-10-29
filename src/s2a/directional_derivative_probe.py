"""Fixed, charged direction probes with a mandatory independent anchor repeat.

This is a finite diagnostic, not an optimizer. A repeated anchor is never served
from the grid cache. Differences use nominal h, and all original audit/cap labels
are retained even when a numerical response is unsafe or misses its pose target.
"""

from copy import deepcopy

import numpy as np

from s2a.geometry import finite, pose_error, validate_transform
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints

STEP_SIZES = (1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8)
SAFETY_NAMES = ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")


def _canonical(value):
    """Use JSON numeric values, without claiming dtype or hidden empty axes."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _numeric_array(value):
    if not isinstance(value, list):
        return None
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    return array if array.dtype.kind in "biuf" else None


def _repeat_comparison(initial, repeated, repeated_index):
    arrays = {}
    structure_equal = scalars_equal = True

    def visit(left, right, path):
        nonlocal structure_equal, scalars_equal
        a, b = _numeric_array(left), _numeric_array(right)
        if a is not None or b is not None:
            compatible = a is not None and b is not None and a.shape == b.shape
            exact = bool(compatible and np.array_equal(a, b))
            structure_equal = structure_equal and compatible
            difference = None
            if compatible and np.isfinite(a).all() and np.isfinite(b).all():
                difference = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
            arrays[path] = dict(
                shape=None if a is None else list(a.shape),
                repeated_shape=None if b is None else list(b.shape),
                exact=exact,
                max_abs_difference=difference,
            )
        elif isinstance(left, dict) and isinstance(right, dict):
            structure_equal = structure_equal and set(left) == set(right)
            for key in sorted(set(left) & set(right)):
                visit(left[key], right[key], f"{path}.{key}" if path else key)
        elif isinstance(left, list) and isinstance(right, list):
            structure_equal = structure_equal and len(left) == len(right)
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f"{path}[{index}]")
        else:
            if isinstance(left, (dict, list)) or isinstance(right, (dict, list)):
                structure_equal = False
            scalars_equal = scalars_equal and type(left) is type(right) and left == right

    visit(_canonical(initial), _canonical(repeated), "")
    maximum = max(
        (row["max_abs_difference"] for row in arrays.values() if row["max_abs_difference"] is not None), default=0.0
    )
    exact = bool(structure_equal and scalars_equal and all(row["exact"] for row in arrays.values()))
    return dict(
        initial_response_index=0,
        repeated_response_index=repeated_index,
        comparison_scope="Canonical JSON numeric arrays and scalars; no dtype or hidden empty-array dimension claim",
        exact=exact,
        structure_equal=bool(structure_equal),
        scalars_equal=bool(scalars_equal),
        arrays=arrays,
        max_abs_difference=maximum,
    )


def probe_directional_derivatives(
    predict,
    previous,
    anchor,
    directions,
    interface,
    lower,
    upper,
    desired,
    site_from_tcp,
    *,
    past_inner_error_rad,
    terminal_normals,
    source_stages=(24, 25, 26),
    step_sizes=STEP_SIZES,
    max_responses=80,
):
    """Evaluate the entire declared grid, then charge a second anchor response.

    Smaller horizons and grids support component checks. The budget must cover
    the worst-case call count including the forced repeat before any prediction.
    Command admission uses original box/rate slacks, with their 1e-12 tolerance.
    Both accepted and rejected command requests retain their actual displacement.
    """
    previous = finite(previous, (6,)).copy()
    anchor, directions = finite(anchor).copy(), finite(directions).copy()
    if anchor.ndim != 2 or anchor.shape[1] != 6 or not 1 <= len(anchor) <= 16:
        raise ValueError("One to sixteen explicit six-axis anchor commands required")
    h = len(anchor)
    if directions.ndim != 3 or directions.shape[1:] != anchor.shape or not len(directions):
        raise ValueError("One or more complete H by six actual saved directions required")
    stages, steps = list(source_stages), finite(step_sizes).copy()
    if (
        len(stages) != len(directions)
        or any(type(stage) is not int or stage < 0 for stage in stages)
        or len(set(stages)) != len(stages)
        or steps.ndim != 1
        or not len(steps)
        or np.any(steps <= 0)
        or np.any(np.diff(steps) >= 0)
    ):
        raise ValueError("Distinct source stages and positive strictly decreasing nominal step sizes required")
    planned = 2 + 2 * len(directions) * len(steps)
    if type(max_responses) is not int or not planned <= max_responses <= 80:
        raise ValueError("Explicit budget must cover the complete fixed grid and mandatory repeat, at most80calls")
    scales = np.max(np.abs(directions), axis=(1, 2))
    if np.any(scales == 0):
        raise ValueError("A saved direction must have a nonzero infinity norm")
    units = finite(directions / scales[:, None, None])
    flat = units.reshape(len(units), -1)
    pairwise_linf = finite(np.max(np.abs(flat[:, None] - flat[None, :]), axis=-1))
    norms = np.linalg.norm(flat, axis=1)
    pairwise_cosine = finite((flat @ flat.T) / np.outer(norms, norms))
    interface = deepcopy(interface)
    input_min, input_max = finite(interface.input_min, (6,)).copy(), finite(interface.input_max, (6,)).copy()
    lower, upper = finite(lower).copy(), finite(upper).copy()
    desired, tool = validate_transform(desired).copy(), validate_transform(site_from_tcp).copy()
    profile, normals = finite(past_inner_error_rad).copy(), finite(terminal_normals).copy()
    if (
        np.any(input_min >= input_max)
        or lower.ndim != 1
        or not len(lower)
        or upper.shape != lower.shape
        or np.any(lower >= upper)
        or profile.shape != (25, len(lower))
        or np.any(profile < 0)
        or np.any(np.diff(profile, axis=0) < 0)
        or normals.ndim != 2
        or normals.shape[1] != len(lower)
        or desired.shape != (h, 4, 4)
        or tool.shape != (4, 4)
    ):
        raise ValueError("Valid original bounds, transforms, history profile and explicit terminal normals required")

    def admission(commands):
        change = np.diff(np.vstack((previous, commands)), axis=0)
        slacks = dict(
            box_lower=finite(commands - input_min),
            box_upper=finite(input_max - commands),
            rate_lower=finite(0.25 + change),
            rate_upper=finite(0.25 - change),
        )
        reasons = [name for name, slack in slacks.items() if np.any(slack < -1e-12)]
        return dict(admitted=not reasons, tolerance=1e-12, **slacks, reasons=reasons)

    if not admission(anchor)["admitted"]:
        raise ValueError("The fixed anchor must pass original command box/rate admission without clipping")
    ledger, entries, cache = [], [], {}

    def evaluate(commands):
        rollout = deepcopy(predict(commands.copy()))
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
        poses = validate_transform(rollout["pose"])
        if poses.shape != (h, 4, 4):
            raise ValueError("Every predicted endpoint requires its own FK pose")
        errors = pose_error(desired, poses @ tool)
        normalized = finite(
            np.stack(
                (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2), axis=1
            )
        )
        failures = np.flatnonzero(~np.all(normalized <= 1.0, axis=1))
        prefix = int(failures[0]) if len(failures) else h
        safety = np.concatenate([audit["groups"][name]["slack"].ravel() for name in SAFETY_NAMES])
        return dict(
            commands=commands.copy(),
            rollout=rollout,
            audit=audit,
            tcp_errors=errors,
            normalized_squared_errors=normalized,
            safety_slack=safety,
            prefix_length=prefix,
            maximum_normalized_squared_error=float(np.max(normalized)),
            maximum_pose_cap_violation=float(max(0.0, np.max(np.sqrt(normalized)) - 1.0)),
            pose_loss=float(np.sum(normalized)),
            feasible=bool(audit["feasible"] and prefix == h),
        )

    def request(commands, role, *, direction_index=None, step_index=None, sign=None):
        commands = finite(commands, (h, 6)).copy()
        step = None if step_index is None else float(steps[step_index])
        nominal = np.zeros_like(anchor) if step is None else sign * step * units[direction_index]
        actual = commands - anchor
        error = float(np.max(np.abs(actual - nominal)))
        key = commands.tobytes()
        gate = admission(commands)
        row = dict(
            request_index=len(ledger),
            role=role,
            direction_index=direction_index,
            step_index=step_index,
            source_stage=None if direction_index is None else stages[direction_index],
            step=step,
            sign=sign,
            commands=commands,
            command_key=key.hex(),
            admission=gate,
            actual_displacement=actual,
            nominal_displacement=nominal.copy(),
            displacement_error_inf=error,
            relative_displacement_error_inf=None if step is None else error / step,
            zero_displacement=bool(not np.any(actual != 0)),
            outcome=None,
            response_index=None,
            charged=False,
            calls_before=len(entries),
            calls_after=len(entries),
            repeat_of=0 if role == "anchor_repeat" else None,
        )
        ledger.append(row)
        if not gate["admitted"]:
            row["outcome"] = "command_rejected"
        elif role != "anchor_repeat" and key in cache:
            row.update(outcome="cached", response_index=cache[key])
        else:
            if len(entries) >= max_responses:
                raise RuntimeError("Prevalidated fixed diagnostic call budget was exceeded")
            index = len(entries)
            entries.append(evaluate(commands))
            cache.setdefault(key, index)
            row.update(
                outcome="repeated" if role == "anchor_repeat" else "computed",
                response_index=index,
                charged=True,
                calls_after=len(entries),
            )
        return row

    request(anchor, "anchor")
    grid_pairs = []
    for direction_index, unit in enumerate(units):
        for step_index, step in enumerate(steps):
            plus = request(anchor + step * unit, "grid", direction_index=direction_index, step_index=step_index, sign=1)
            minus = request(
                anchor - step * unit, "grid", direction_index=direction_index, step_index=step_index, sign=-1
            )
            grid_pairs.append((direction_index, step_index, plus, minus))
    request(anchor, "anchor_repeat")

    def difference(plus, minus, kind, step):
        sides = (plus, minus) if kind == "central" else (plus,) if kind == "forward" else (minus,)
        reason = (
            "command_rejected"
            if any(not side["admission"]["admitted"] for side in sides)
            else ("zero_displacement" if any(side["zero_displacement"] for side in sides) else None)
        )
        values = dict(valid=reason is None, reason=reason, safety_slack=None, normalized_squared_errors=None)
        if reason is None:
            first = entries[0] if kind == "backward" else entries[plus["response_index"]]
            second = entries[0] if kind == "forward" else entries[minus["response_index"]]
            denominator = 2 * step if kind == "central" else step
            for name in ("safety_slack", "normalized_squared_errors"):
                values[name] = finite((first[name] - second[name]) / denominator)
        return values

    differences = []
    for direction_index, step_index, plus, minus in grid_pairs:
        step = float(steps[step_index])
        differences.append(
            dict(
                direction_index=direction_index,
                source_stage=stages[direction_index],
                step_index=step_index,
                step=step,
                plus_request_index=plus["request_index"],
                minus_request_index=minus["request_index"],
                **{kind: difference(plus, minus, kind, step) for kind in ("forward", "backward", "central")},
            )
        )
    repeatability = _repeat_comparison(entries[0], entries[-1], len(entries) - 1)
    return dict(
        method="fixed_directional_derivative_probe",
        horizon=h,
        previous_command=previous,
        input_min=input_min,
        input_max=input_max,
        anchor_commands=anchor,
        directions=directions,
        direction_scales=scales,
        unit_directions=units,
        source_stages=stages,
        step_sizes=steps.tolist(),
        pairwise_direction_linf=pairwise_linf,
        pairwise_direction_cosine=pairwise_cosine,
        max_responses=max_responses,
        planned_request_count=planned,
        response_evaluations=len(entries),
        unique_command_count=len(cache),
        total_iterations=0,
        request_ledger=ledger,
        evaluated_candidates=entries,
        differences=differences,
        repeatability=repeatability,
        derivative_reliability_limited=not repeatability["exact"],
        control_qualified=False,
    )
