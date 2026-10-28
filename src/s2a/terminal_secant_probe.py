"""Finite terminal-row secant fitting and an independent direction geometry.

This component prepares a fixed diagnostic. Its fitted row is a finite-scale
model and does not establish a limiting derivative or control qualification.
"""

from copy import deepcopy

import numpy as np

from s2a.directional_derivative_probe import _canonical, _repeat_comparison
from s2a.geometry import finite, pose_error, validate_transform
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints

STEP_SIZES = (1e-3, 7.5e-4, 5e-4, 2.5e-4, 1e-4)
FIT_STEP_SIZES = (1e-3, 5e-4)
DIRECTION_NAMES = ("orthogonal", "source_plus_orthogonal", "source_minus_orthogonal")
COMMAND_GROUPS = ("box_lower", "box_upper", "rate_lower", "rate_upper")
SAFETY_NAMES = ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _admission(commands, previous, lower, upper):
    increments = np.diff(np.vstack((previous, commands)), axis=0)
    slacks = dict(
        box_lower=finite(commands - lower),
        box_upper=finite(upper - commands),
        rate_lower=finite(0.25 + increments),
        rate_upper=finite(0.25 - increments),
    )
    reasons = [name for name in COMMAND_GROUPS if np.any(slacks[name] < -1e-12)]
    return dict(admitted=not reasons, tolerance=1e-12, **slacks, reasons=reasons)


def build_terminal_secant_geometry(
    anchor_commands,
    previous_command,
    input_min,
    input_max,
    source_unit_directions,
    *,
    seed=42,
    svd_rcond=1e-12,
    box_guard=1e-3,
    rate_guard=2e-3,
    step_sizes=STEP_SIZES,
):
    """Project one Gaussian draw and preflight every complete grid command.

    No response is evaluated. Source rows retain their unit-infinity scaling;
    command rows retain their original signed gradients. A failed grid is
    returned in full with ``admitted=False`` and is never clipped or retried.
    """
    anchor = finite(anchor_commands).copy()
    _require(anchor.ndim == 2 and anchor.shape[1] == 6 and 1 <= len(anchor) <= 16, "Complete H by six anchor")
    horizon, width = len(anchor), anchor.size
    previous = finite(previous_command, (6,)).copy()
    lower, upper = finite(input_min, (6,)).copy(), finite(input_max, (6,)).copy()
    sources = finite(source_unit_directions, (3, horizon, 6)).copy()
    steps = finite(step_sizes).copy()
    _require(np.all(lower < upper), "Strict original input ranges")
    _require(np.all(np.max(np.abs(sources), axis=(1, 2)) == 1.0), "Three source directions normalized in infinity norm")
    _require(type(seed) is int and seed >= 0, "Explicit nonnegative integer seed")
    _require(np.isfinite(svd_rcond) and 0 < svd_rcond < 1, "Finite relative SVD cutoff")
    _require(
        np.isfinite(box_guard) and box_guard > 0 and np.isfinite(rate_guard) and rate_guard > 0, "Positive guard bands"
    )
    _require(
        steps.ndim == 1 and len(steps) > 0 and np.all(steps > 0) and np.all(np.diff(steps) < 0),
        "Positive descending steps",
    )
    anchor_admission = _admission(anchor, previous, lower, upper)
    _require(anchor_admission["admitted"], "Original anchor must satisfy box/rate admission without clipping")
    identity = np.eye(width)
    difference = identity - np.eye(width, k=-6)
    gradients = dict(box_lower=identity, box_upper=-identity, rate_lower=difference, rate_upper=-difference)
    rows = [row.copy() for row in sources.reshape(3, width)]
    row_metadata = [dict(kind="source_direction", source_direction_index=index) for index in range(3)]
    for group in COMMAND_GROUPS:
        guard = box_guard if group.startswith("box") else rate_guard
        slacks = anchor_admission[group].ravel()
        for axis in np.flatnonzero(slacks <= guard):
            rows.append(gradients[group][axis].copy())
            row_metadata.append(
                dict(kind="command_guard", group=group, axis=int(axis), slack=float(slacks[axis]), guard=float(guard))
            )
    projection_rows = np.asarray(rows)
    _, singular_values, right_vectors = np.linalg.svd(projection_rows, full_matrices=False)
    cutoff = float(svd_rcond * singular_values[0])
    rank = int(np.count_nonzero(singular_values > cutoff))
    _require(rank < width, "Command guards and source directions exhaust the projection nullspace")
    gaussian = np.random.default_rng(seed).standard_normal(width)
    row_basis = right_vectors[:rank]
    projected = finite(gaussian - row_basis.T @ (row_basis @ gaussian))
    projected_norm, gaussian_norm = float(np.linalg.norm(projected)), float(np.linalg.norm(gaussian))
    _require(projected_norm > 1e-10, "Gaussian projection norm must exceed1e-10")
    v = projected / projected_norm
    sign_candidates = np.flatnonzero(np.abs(v) > svd_rcond)
    _require(len(sign_candidates) > 0, "Projected direction has a resolved sign coordinate")
    sign_index = int(sign_candidates[0])
    sign_multiplier = 1.0 if v[sign_index] > 0 else -1.0
    v *= sign_multiplier
    projection_residual = finite(projection_rows @ v)
    projection_residual_inf = float(np.max(np.abs(projection_residual)))
    _require(projection_residual_inf <= 1e-10, "Projected direction must satisfy all source and guard rows at1e-10")
    e = sources[0].ravel() / np.linalg.norm(sources[0])
    raw_directions = np.asarray((v, e + v, e - v))
    scales = np.max(np.abs(raw_directions), axis=1)
    _require(np.all(scales > 0), "Three nonzero holdout directions")
    directions = finite(raw_directions / scales[:, None]).reshape(3, horizon, 6)
    commands, keys = [], {anchor.tobytes()}
    all_admitted = True
    for direction_index, direction in enumerate(directions):
        for step_index, step in enumerate(steps):
            command = finite(anchor + step * direction)
            delta, nominal = command - anchor, step * direction
            key = command.tobytes()
            admission = _admission(command, previous, lower, upper)
            unique = key not in keys
            zero = not np.any(delta != 0)
            keys.add(key)
            error = float(np.max(np.abs(delta - nominal)))
            all_admitted = all_admitted and admission["admitted"] and unique and not zero
            commands.append(
                dict(
                    grid_index=len(commands),
                    direction_index=direction_index,
                    direction_name=DIRECTION_NAMES[direction_index],
                    step_index=step_index,
                    step=float(step),
                    commands=command.copy(),
                    command_key=key.hex(),
                    actual_displacement=delta.copy(),
                    nominal_displacement=nominal.copy(),
                    displacement_error_inf=error,
                    relative_displacement_error_inf=error / float(step),
                    zero_displacement=bool(zero),
                    unique_command=unique,
                    admission=admission,
                )
            )
    flat = directions.reshape(3, width)
    norms = np.linalg.norm(flat, axis=1)
    source_flat = sources.reshape(3, width)
    source_norms = np.linalg.norm(source_flat, axis=1)
    return dict(
        admitted=bool(all_admitted),
        horizon=horizon,
        anchor_commands=anchor,
        previous_command=previous,
        input_min=lower,
        input_max=upper,
        source_unit_directions=sources,
        source_direction_index=0,
        seed=seed,
        random_generator="numpy.default_rng/PCG64",
        svd_rcond=float(svd_rcond),
        box_guard=float(box_guard),
        rate_guard=float(rate_guard),
        projection_rows=projection_rows,
        projection_row_metadata=row_metadata,
        singular_values=singular_values.copy(),
        singular_cutoff=cutoff,
        projection_rank=rank,
        nullspace_dimension=width - rank,
        gaussian=gaussian.copy(),
        projected_gaussian=projected.copy(),
        projected_norm=projected_norm,
        projected_to_gaussian_norm_ratio=projected_norm / gaussian_norm,
        sign_index=sign_index,
        sign_multiplier=sign_multiplier,
        sign_resolution_threshold=float(svd_rcond),
        e=e.reshape(horizon, 6).copy(),
        v=v.reshape(horizon, 6).copy(),
        projection_residual=projection_residual.copy(),
        projection_residual_inf=projection_residual_inf,
        raw_directions=raw_directions.reshape(3, horizon, 6).copy(),
        direction_scales=scales.copy(),
        directions=directions.copy(),
        direction_names=list(DIRECTION_NAMES),
        pairwise_direction_cosine=(flat @ flat.T) / np.outer(norms, norms),
        source_direction_cosine=(flat @ source_flat.T) / np.outer(norms, source_norms),
        step_sizes=steps.tolist(),
        grid_commands=commands,
        anchor_admission=anchor_admission,
        planned_request_count=2 + len(commands),
        expected_unique_command_count=1 + len(commands),
    )


def fit_terminal_secant_row(
    anchor_commands,
    source_unit_direction,
    original_terminal_row,
    anchor_terminal_slack,
    fit_commands,
    fit_terminal_slacks,
):
    """Fit one rank-one row correction from two actual saved command deltas.

    There is no free intercept, no chosen finite-difference step, and no use of
    future holdout responses. Source identities and the fixed training steps
    must additionally be bound by the caller's evidence registration.
    """
    anchor = finite(anchor_commands).copy()
    _require(anchor.ndim == 2 and anchor.shape[1] == 6 and 1 <= len(anchor) <= 16, "Complete anchor")
    unit = finite(source_unit_direction, anchor.shape).copy()
    row = finite(original_terminal_row, (anchor.size,)).copy()
    initial = float(finite(anchor_terminal_slack, ()))
    commands = finite(fit_commands, (2, *anchor.shape)).copy()
    terminal = finite(fit_terminal_slacks, (2,)).copy()
    for index, step in enumerate(FIT_STEP_SIZES):
        _require(
            commands[index].tobytes() == (anchor + step * unit).tobytes(),
            "Exact positive source commands at the two fixed fitting steps",
        )
    flat = unit.ravel()
    squared_norm = float(flat @ flat)
    _require(squared_norm > 0 and np.max(np.abs(flat)) == 1.0, "Nonzero source unit-infinity direction")
    delta = commands - anchor
    flat_delta = delta.reshape(2, -1)
    coordinate = finite((flat_delta @ flat) / squared_norm)
    _require(
        np.all(coordinate > 0) and coordinate[0] > coordinate[1],
        "Two distinct descending positive source displacements",
    )
    old_change = finite(flat_delta @ row)
    observed = finite(terminal - initial)
    residual = finite(observed - old_change)
    denominator = float(coordinate @ coordinate)
    _require(np.isfinite(denominator) and denominator > 0, "Resolved finite least-squares denominator")
    coefficient = float(finite(float(coordinate @ residual) / denominator, ()))
    correction = finite(coefficient * flat / squared_norm)
    corrected_row = finite(row + correction)
    corrected_change = finite(flat_delta @ corrected_row)
    return dict(
        anchor_commands=anchor,
        source_unit_direction=unit,
        original_terminal_row=row,
        anchor_terminal_slack=initial,
        fit_commands=commands,
        fit_terminal_slacks=terminal,
        fit_step_sizes=list(FIT_STEP_SIZES),
        actual_displacements=delta.copy(),
        source_coordinates=coordinate.copy(),
        original_predicted_changes=old_change.copy(),
        observed_changes=observed.copy(),
        original_residuals=residual.copy(),
        coefficient=coefficient,
        correction_row=correction.copy(),
        corrected_terminal_row=corrected_row.copy(),
        corrected_predicted_changes=corrected_change.copy(),
        corrected_residuals=finite(observed - corrected_change),
        least_squares_denominator=denominator,
        evidence_scope="Two fixed finite positive displacements; no limiting derivative or control qualification",
    )


def _terminal(entry, normals, horizon, joint_count):
    groups = entry["audit"]["groups"]
    arrays = [finite(groups[name]["slack"]) for name in SAFETY_NAMES]
    safety = np.concatenate([value.ravel() for value in arrays])
    _require(np.array_equal(safety, finite(entry["safety_slack"], safety.shape)), "Full saved safety row layout")
    offset = sum(value.size for value in arrays[:-1])
    terminal = finite(arrays[-1], (len(normals),))
    velocity = finite(entry["rollout"]["joint_velocity"], (horizon, joint_count))
    _require(np.array_equal(terminal, -normals @ velocity[-1]), "Saved terminal rows match original endpoint velocity")
    _require(len(terminal) > 0, "Explicit original terminal row zero required")
    return offset, float(terminal[0])


def _clipping_checks(entries, horizon, joint_count):
    checks, original = [], None
    for index, entry in enumerate(entries):
        periods = entry["rollout"]["periods"]
        _require(len(periods) == horizon, "Every complete period retains torque clipping flags")
        flags = []
        for period in periods:
            value = np.asarray(period["clipped"])
            _require(
                value.shape == (25, joint_count) and value.dtype == np.dtype(bool),
                "Original boolean torque clipping arrays",
            )
            flags.append(value.copy())
        current = np.stack(flags)
        if original is None:
            original = current.copy()
        changed = int(np.count_nonzero(current != original))
        checks.append(dict(response_index=index, exact=changed == 0, changed_count=changed, period_count=horizon))
    return checks


def _qualification(predictions, clipping_checks, repeatability, historical_comparison, fit):
    mixed = []
    for index in (1, 2):
        rows = [row for row in predictions if row["direction_index"] == index]
        original = max(abs(row["original_residual"]) for row in rows)
        corrected = max(abs(row["corrected_residual"]) for row in rows)
        original_normalized = max(abs(row["original_residual_over_step"]) for row in rows)
        corrected_normalized = max(abs(row["corrected_residual_over_step"]) for row in rows)
        absolute_passed = corrected <= 0.5 * original
        normalized_passed = corrected_normalized <= 0.5 * original_normalized
        mixed.append(
            dict(
                direction_index=index,
                direction_name=DIRECTION_NAMES[index],
                original_max_abs_residual=original,
                corrected_max_abs_residual=corrected,
                original_max_abs_normalized_residual=original_normalized,
                corrected_max_abs_normalized_residual=corrected_normalized,
                absolute_passed=absolute_passed,
                normalized_passed=normalized_passed,
                passed=absolute_passed and normalized_passed,
            )
        )
    pure_difference = max(abs(row["prediction_correction"]) for row in predictions if row["direction_index"] == 0)
    pure = dict(max_abs_prediction_change=pure_difference, tolerance=1e-12, passed=pure_difference <= 1e-12)
    clipped = all(row["exact"] for row in clipping_checks)
    repeat_exact, historical_exact = repeatability["exact"], historical_comparison["comparison"]["exact"]
    extrapolation = (
        all(row["passed"] for row in mixed) and pure["passed"] and clipped and repeat_exact and historical_exact
    )
    margin = fit["anchor_terminal_slack"] - 1e-8
    maximum = max(abs(row["corrected_residual"]) for row in predictions)
    return dict(
        mixed_direction_checks=mixed,
        pure_direction_check=pure,
        all_clipping_equal=clipped,
        repeat_exact=repeat_exact,
        historical_exact=historical_exact,
        improvement_factor=0.5,
        safety_buffer=1e-8,
        safety_available_margin=margin,
        max_abs_corrected_residual=maximum,
        extrapolation_passed=bool(extrapolation),
        safety_accuracy_passed=bool(margin >= 0 and maximum <= margin),
    )


def probe_terminal_secant(
    predict,
    previous,
    anchor,
    interface,
    lower,
    upper,
    desired,
    site_from_tcp,
    *,
    past_inner_error_rad,
    terminal_normals,
    geometry,
    fit,
    historical_anchor,
    max_responses=17,
):
    """Evaluate one fully preflighted17-call grid without solving an optimization.

    All15 commands must pass original box/rate admission and have distinct bytes
    before the first callback. The final anchor is the only repeated command;
    it always incurs another complete callback. No cache or early witness exit
    changes this fixed grid, and no holdout response changes the fitted row.
    """
    _require(type(max_responses) is int and max_responses == 17, "The fixed grid requires exactly17 allowed calls")
    interface = deepcopy(interface)
    rebuilt = build_terminal_secant_geometry(
        anchor, previous, interface.input_min, interface.input_max, geometry["source_unit_directions"]
    )
    _require(
        _canonical(rebuilt) == _canonical(geometry),
        "Frozen complete geometry matches the original inputs and fixed construction",
    )
    geometry = rebuilt
    _require(geometry["admitted"], "Entire fixed grid must pass original admission with distinct nonzero commands")
    _require(geometry["planned_request_count"] == 17, "Complete fixed three-direction/five-step grid")
    anchor, previous = geometry["anchor_commands"], geometry["previous_command"]
    horizon = len(anchor)
    lower, upper = finite(lower).copy(), finite(upper).copy()
    desired, tool = validate_transform(desired).copy(), validate_transform(site_from_tcp).copy()
    profile, normals = finite(past_inner_error_rad).copy(), finite(terminal_normals).copy()
    _require(
        lower.ndim == 1 and len(lower) > 0 and upper.shape == lower.shape and np.all(lower < upper),
        "Original joint limits",
    )
    joint_count = len(lower)
    _require(
        desired.shape == (horizon, 4, 4)
        and tool.shape == (4, 4)
        and profile.shape == (25, joint_count)
        and np.all(profile >= 0)
        and np.all(np.diff(profile, axis=0) >= 0)
        and normals.ndim == 2
        and normals.shape[1] == joint_count
        and len(normals) > 0,
        "Complete original transforms, history error profile and terminal rows",
    )
    _require(
        finite(fit["anchor_commands"], anchor.shape).tobytes() == anchor.tobytes(),
        "Fit and geometry use identical anchor bytes",
    )
    _require(
        finite(fit["source_unit_direction"], anchor.shape).tobytes() == geometry["source_unit_directions"][0].tobytes(),
        "Fit and geometry use the same original source direction",
    )
    fitted = fit_terminal_secant_row(
        anchor,
        geometry["source_unit_directions"][0],
        fit["original_terminal_row"],
        fit["anchor_terminal_slack"],
        fit["fit_commands"],
        fit["fit_terminal_slacks"],
    )
    _require(_canonical(fitted) == _canonical(fit), "Complete frozen fit agrees with actual saved displacement algebra")
    historical = deepcopy(historical_anchor)
    _require(
        finite(historical["commands"], anchor.shape).tobytes() == anchor.tobytes(),
        "Historical complete anchor has the identical command bytes",
    )
    historical_offset, historical_terminal = _terminal(historical, normals, horizon, joint_count)
    _require(
        historical_terminal == fitted["anchor_terminal_slack"],
        "Fit baseline is the historical complete anchor terminal",
    )
    entries, ledger = [], []

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
        _require(poses.shape == (horizon, 4, 4), "Every predicted endpoint requires its own FK pose")
        errors = pose_error(desired, poses @ tool)
        normalized = finite(
            np.stack(
                (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2), axis=1
            )
        )
        failures = np.flatnonzero(np.any(normalized > 1.0, axis=1))
        prefix = int(failures[0]) if len(failures) else horizon
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
            feasible=bool(audit["feasible"] and prefix == horizon),
        )

    def request(commands, role, grid=None):
        commands = commands.copy()
        actual = commands - anchor
        nominal = np.zeros_like(anchor) if grid is None else grid["nominal_displacement"].copy()
        error = float(np.max(np.abs(actual - nominal)))
        index = len(entries)
        _require(index < max_responses, "No response beyond the fixed call budget")
        entry = evaluate(commands)
        entries.append(entry)
        ledger.append(
            dict(
                request_index=len(ledger),
                role=role,
                direction_index=None if grid is None else grid["direction_index"],
                direction_name=None if grid is None else grid["direction_name"],
                step_index=None if grid is None else grid["step_index"],
                step=None if grid is None else grid["step"],
                commands=commands.copy(),
                command_key=commands.tobytes().hex(),
                actual_displacement=actual.copy(),
                nominal_displacement=nominal,
                displacement_error_inf=error,
                relative_displacement_error_inf=None if grid is None else error / grid["step"],
                admission=deepcopy(geometry["anchor_admission"] if grid is None else grid["admission"]),
                zero_displacement=bool(not np.any(actual != 0)),
                response_index=index,
                calls_before=index,
                calls_after=len(entries),
                charged=True,
                outcome="repeated" if role == "anchor_repeat" else "computed",
                repeat_of=0 if role == "anchor_repeat" else None,
            )
        )

    request(anchor, "anchor")
    for grid in geometry["grid_commands"]:
        request(grid["commands"], "grid", grid)
    request(anchor, "anchor_repeat")
    terminal_values = []
    for entry in entries:
        offset, terminal = _terminal(entry, normals, horizon, joint_count)
        _require(offset == historical_offset, "Historical and every new call use identical terminal row positions")
        terminal_values.append(terminal)
    predictions = []
    for row in ledger[1:-1]:
        displacement = row["actual_displacement"].ravel()
        original = float(finite(fitted["original_terminal_row"] @ displacement, ()))
        corrected = float(finite(fitted["corrected_terminal_row"] @ displacement, ()))
        observed = float(finite(terminal_values[row["response_index"]] - terminal_values[0], ()))
        old_residual, new_residual = observed - original, observed - corrected
        predictions.append(
            dict(
                request_index=row["request_index"],
                direction_index=row["direction_index"],
                direction_name=row["direction_name"],
                step_index=row["step_index"],
                step=row["step"],
                original_predicted_change=original,
                corrected_predicted_change=corrected,
                observed_change=observed,
                original_residual=old_residual,
                corrected_residual=new_residual,
                original_residual_over_step=old_residual / row["step"],
                corrected_residual_over_step=new_residual / row["step"],
                prediction_correction=corrected - original,
            )
        )
    repeatability = _repeat_comparison(entries[0], entries[-1], 16)
    comparison = _repeat_comparison(historical, entries[0], 0)
    del comparison["initial_response_index"], comparison["repeated_response_index"]
    historical_comparison = dict(historical_response_index=0, new_response_index=0, comparison=comparison)
    clipping_checks = _clipping_checks(entries, horizon, joint_count)
    qualification = _qualification(predictions, clipping_checks, repeatability, historical_comparison, fitted)
    return dict(
        method="fixed_terminal_secant_probe",
        horizon=horizon,
        previous_command=previous.copy(),
        input_min=geometry["input_min"].copy(),
        input_max=geometry["input_max"].copy(),
        anchor_commands=anchor.copy(),
        source_unit_directions=geometry["source_unit_directions"].copy(),
        step_sizes=list(STEP_SIZES),
        direction_names=list(DIRECTION_NAMES),
        directions=geometry["directions"].copy(),
        geometry=geometry,
        fit=fitted,
        max_responses=max_responses,
        planned_request_count=17,
        response_evaluations=len(entries),
        unique_command_count=len({row["command_key"] for row in ledger}),
        total_iterations=0,
        request_ledger=ledger,
        evaluated_candidates=entries,
        terminal_row_index=historical_offset,
        predictions=predictions,
        repeatability=repeatability,
        historical_anchor_comparison=historical_comparison,
        clipping_checks=clipping_checks,
        qualification=qualification,
        extrapolation_passed=qualification["extrapolation_passed"],
        safety_accuracy_passed=qualification["safety_accuracy_passed"],
        derivative_reliability_limited=not repeatability["exact"],
        control_qualified=False,
    )
