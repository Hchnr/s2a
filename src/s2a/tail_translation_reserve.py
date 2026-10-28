"""Fixed last-period translation probes under the original preview constraints.

Every admitted command gets a complete response, including the paid final
anchor repeat. Restoration labels remain separate from the original full-H12
pose/joint witness and never establish physical or task qualification.
"""

from copy import deepcopy

import numpy as np

from s2a.directional_derivative_probe import _canonical, _repeat_comparison
from s2a.geometry import finite, pose_error, validate_transform
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints
from s2a.terminal_secant_probe import _clipping_checks, _terminal

SIGNED_STEPS = (0.001, -0.001, 0.0005, -0.0005)
AXIS_NAMES = ("x", "y", "z")
SAFETY_NAMES = ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")
SAFETY_BUFFER = 1e-8
REQUIRED_PREFIX = 9
RESERVE_FACTOR = 10.0


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _admission(commands, previous, lower, upper):
    change = np.diff(np.vstack((previous, commands)), axis=0)
    groups = dict(
        box_lower=commands - lower, box_upper=upper - commands, rate_lower=0.25 + change, rate_upper=0.25 - change
    )
    reasons = [name for name, slack in groups.items() if np.any(slack < -1e-12)]
    return dict(admitted=not reasons, tolerance=1e-12, **groups, reasons=reasons)


def build_tail_translation_grid(anchor_commands, previous_command, input_min, input_max):
    """Construct and preflight all twelve fixed commands without a predictor."""
    anchor = finite(anchor_commands, (12, 6)).copy()
    previous = finite(previous_command, (6,)).copy()
    lower, upper = finite(input_min, (6,)).copy(), finite(input_max, (6,)).copy()
    _require(np.all(lower < upper), "Strict original command bounds")
    anchor_admission = _admission(anchor, previous, lower, upper)
    rows, seen = [], {anchor.tobytes()}
    for axis, name in enumerate(AXIS_NAMES):
        for step_index, step in enumerate(SIGNED_STEPS):
            command, nominal = anchor.copy(), np.zeros_like(anchor)
            command[11, axis] += step
            nominal[11, axis] = step
            command = finite(command, anchor.shape)
            actual = command - anchor
            key = command.tobytes()
            error = float(np.max(np.abs(actual - nominal)))
            row = dict(
                grid_index=len(rows),
                axis_index=axis,
                axis_name=name,
                step_index=step_index,
                signed_step=step,
                commands=command.copy(),
                command_key=key.hex(),
                actual_displacement=actual,
                nominal_displacement=nominal,
                displacement_error_inf=error,
                relative_displacement_error_inf=error / abs(step),
                actual_displacement_linf=float(np.max(np.abs(actual))),
                zero_displacement=bool(not np.any(actual != 0)),
                unique_command=key not in seen,
                admission=_admission(command, previous, lower, upper),
            )
            rows.append(row)
            seen.add(key)
    admitted = anchor_admission["admitted"] and all(
        row["admission"]["admitted"] and row["unique_command"] and not row["zero_displacement"] for row in rows
    )
    return dict(
        horizon=12,
        tail_period_index=11,
        axis_names=list(AXIS_NAMES),
        signed_steps=list(SIGNED_STEPS),
        anchor_commands=anchor,
        previous_command=previous,
        input_min=lower,
        input_max=upper,
        anchor_admission=anchor_admission,
        grid_commands=rows,
        admitted=bool(admitted),
        planned_request_count=14,
        expected_unique_command_count=13,
    )


def _causal_payloads(rollout):
    """Keep every saved prefix period field, plus both stored goal rotations."""
    periods = rollout["periods"]
    _require(len(periods) == 12, "Every full native period must be retained")
    for period in periods:
        for key in ("joint_position", "joint_velocity", "requested_torque", "applied_torque"):
            finite(period[key], (25, 7))
        for key in ("mass_change", "jacobian_change"):
            finite(period[key], (25,))
        finite(period["endpoint_twist"], (6,))
        # Preserve the native float32-derived goal without adding a new
        # orthogonality tolerance to the fixed controller's saved output.
        finite(period["goal"], (4, 4))
        finite(period["endpoint_pose"], (4, 4))
        _require(
            type(period["update_matrices"]) is bool and period["update_matrices"] and period["dynamics_queries"] == 26,
            "Original state-updated full native period",
        )
    prefix = dict(periods=deepcopy(periods[:11]))
    for key in ("inner_joint_position", "inner_joint_velocity"):
        prefix[key] = finite(rollout[key], (300, 7))[:275].copy()
    for key in ("joint_position", "joint_velocity"):
        prefix[key] = finite(rollout[key], (12, 7))[:11].copy()
    for key in ("pose", "goals"):
        prefix[key] = finite(rollout[key], (12, 4, 4))[:11].copy()
    prefix["twist"] = finite(rollout["twist"], (12, 6))[:11].copy()
    rotation = dict(
        goals=finite(rollout["goals"], (12, 4, 4))[:, :3, :3].copy(),
        period_goals=np.stack([finite(period["goal"], (4, 4))[:3, :3] for period in periods]),
    )
    return prefix, rotation


def _causal_checks(entries):
    initial_prefix, initial_rotations = _causal_payloads(entries[0]["rollout"])
    checks = []
    for index, entry in enumerate(entries):
        prefix, rotations = _causal_payloads(entry["rollout"])
        first = _repeat_comparison(initial_prefix, prefix, index)
        second = _repeat_comparison(initial_rotations, rotations, index)
        checks.append(
            dict(
                response_index=index,
                prefix_period_count=11,
                prefix_inner_count=275,
                prefix_endpoint_count=11,
                prefix_comparison=first,
                rotation_goal_comparison=second,
                prefix_exact=first["exact"],
                rotation_goals_exact=second["exact"],
                passed=first["exact"] and second["exact"],
            )
        )
    return checks


def _select(entries, ledger, terminal_values, causal, historical, repeat, reserve_target):
    candidates = []
    for index in range(1, 13):
        entry, request = entries[index], ledger[index]
        safety = entry["safety_slack"]
        safety_min = float(np.min(safety))
        gates = dict(
            joint_audit=entry["audit"]["feasible"],
            safety_buffer=bool(np.all(safety >= SAFETY_BUFFER)),
            prefix9=bool(np.all(entry["normalized_squared_errors"][:REQUIRED_PREFIX] <= 1.0)),
            causal_prefix=causal[index]["prefix_exact"],
            rotation_goals=causal[index]["rotation_goals_exact"],
            historical_anchor=historical["comparison"]["exact"],
            anchor_repeat=repeat["exact"],
        )
        candidates.append(
            dict(
                request_index=index,
                response_index=index,
                **{
                    key: request[key]
                    for key in ("axis_index", "axis_name", "step_index", "signed_step", "actual_displacement_linf")
                },
                terminal=terminal_values[index],
                terminal_reserve=terminal_values[index] - SAFETY_BUFFER,
                terminal_improvement=terminal_values[index] - terminal_values[0],
                other_safety_min=float(np.min(safety[:-1])),
                safety_min=safety_min,
                joint_feasible=gates["joint_audit"],
                safety_buffer_satisfied=gates["safety_buffer"],
                prefix9_preserved=gates["prefix9"],
                causal_exact=causal[index]["passed"],
                historical_exact=gates["historical_anchor"],
                repeat_exact=gates["anchor_repeat"],
                restoration_eligible=all(gates.values()),
                rejection_reasons=[key for key, value in gates.items() if not value],
            )
        )
    eligible = sorted(
        (row for row in candidates if row["restoration_eligible"]),
        key=lambda row: (-row["terminal"], row["actual_displacement_linf"], row["request_index"]),
    )
    ranking = [row["response_index"] for row in eligible]
    best = eligible[0] if eligible and eligible[0]["terminal"] > terminal_values[0] else None
    if best is None:
        return candidates, ranking, None, None, None, False
    index = best["response_index"]
    selected = deepcopy(entries[index])
    sufficient = bool(best["terminal_reserve"] >= reserve_target)
    summary = dict(
        **{
            key: best[key]
            for key in (
                "response_index",
                "request_index",
                "terminal",
                "terminal_reserve",
                "terminal_improvement",
                "other_safety_min",
                "safety_min",
                "actual_displacement_linf",
            )
        },
        prefix_length=selected["prefix_length"],
        tcp_errors=selected["tcp_errors"].copy(),
        normalized_squared_errors=selected["normalized_squared_errors"].copy(),
        feasible=selected["feasible"],
        restoration_eligible=True,
        sufficient_reserve_restored=sufficient,
    )
    return candidates, ranking, index, selected, summary, sufficient


def probe_tail_translation_reserve(
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
    grid,
    historical_anchor,
    prior_max_abs_residual,
    max_responses=14,
):
    """Run all fourteen prevalidated responses, then label and rank restoration.

    The callback must reset to the same causal initial state on every call.
    The complete saved prefix/rotation comparisons expose violations; they do
    not repair them or stop the fixed experiment early.
    """
    _require(type(max_responses) is int and max_responses == 14, "Exactly fourteen response calls are allowed")
    interface = deepcopy(interface)
    rebuilt = build_tail_translation_grid(anchor, previous, interface.input_min, interface.input_max)
    _require(_canonical(rebuilt) == _canonical(grid), "Complete fixed grid must match original input generation")
    grid = rebuilt
    _require(grid["admitted"], "Anchor and all twelve distinct nonzero commands must pass preflight")
    anchor, previous = grid["anchor_commands"], grid["previous_command"]
    lower, upper = finite(lower, (7,)).copy(), finite(upper, (7,)).copy()
    _require(np.all(lower < upper), "Original seven-joint bounds")
    desired, tool = validate_transform(desired).copy(), validate_transform(site_from_tcp).copy()
    profile, normals = finite(past_inner_error_rad, (25, 7)).copy(), finite(terminal_normals, (1, 7)).copy()
    _require(
        desired.shape == (12, 4, 4) and tool.shape == (4, 4),
        "All twelve original reference endpoints and TCP transform",
    )
    _require(
        np.all(profile >= 0) and np.all(np.diff(profile, axis=0) >= 0),
        "Original nonnegative nondecreasing history profile",
    )
    _require(np.array_equal(normals[0], np.eye(7)[6]), "Original e7 terminal row")
    residual = float(finite(prior_max_abs_residual, ()))
    target = float(finite(RESERVE_FACTOR * residual, ()))
    _require(residual >= 0, "Nonnegative source-bound prior residual")
    historical = deepcopy(historical_anchor)
    _require(
        finite(historical["commands"], (12, 6)).tobytes() == anchor.tobytes(), "Historical full anchor command identity"
    )
    historical_offset, _ = _terminal(historical, normals, 12, 7)
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
        _require(poses.shape == (12, 4, 4), "Complete H12 endpoint poses")
        errors = pose_error(desired, poses @ tool)
        normalized = finite(
            np.stack(
                (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2), axis=1
            )
        )
        failures = np.flatnonzero(np.any(normalized > 1.0, axis=1))
        prefix = int(failures[0]) if len(failures) else 12
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
            feasible=bool(audit["feasible"] and prefix == 12),
        )

    def request(commands, role, row=None):
        commands = commands.copy()
        actual = commands - anchor
        nominal = np.zeros_like(anchor) if row is None else row["nominal_displacement"].copy()
        error = float(np.max(np.abs(actual - nominal)))
        index = len(entries)
        _require(index < max_responses, "No extra response outside the registered grid")
        entries.append(evaluate(commands))
        ledger.append(
            dict(
                request_index=index,
                role=role,
                **{
                    key: None if row is None else row[key]
                    for key in ("axis_index", "axis_name", "step_index", "signed_step")
                },
                commands=commands.copy(),
                command_key=commands.tobytes().hex(),
                actual_displacement=actual.copy(),
                nominal_displacement=nominal,
                displacement_error_inf=error,
                relative_displacement_error_inf=None if row is None else error / abs(row["signed_step"]),
                actual_displacement_linf=float(np.max(np.abs(actual))),
                admission=deepcopy(grid["anchor_admission"] if row is None else row["admission"]),
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
    for row in grid["grid_commands"]:
        request(row["commands"], "grid", row)
    request(anchor, "anchor_repeat")
    terminal_values = []
    for entry in entries:
        offset, terminal = _terminal(entry, normals, 12, 7)
        _require(offset == historical_offset, "Consistent full safety-vector terminal row")
        terminal_values.append(terminal)
    repeat = _repeat_comparison(entries[0], entries[-1], 13)
    comparison = _repeat_comparison(historical, entries[0], 0)
    del comparison["initial_response_index"], comparison["repeated_response_index"]
    history = dict(historical_response_index=0, new_response_index=0, comparison=comparison)
    causal, clipping = _causal_checks(entries), _clipping_checks(entries, 12, 7)
    candidates, ranking, selected_index, selected, summary, sufficient = _select(
        entries, ledger, terminal_values, causal, history, repeat, target
    )
    return dict(
        method="fixed_tail_translation_reserve",
        horizon=12,
        tail_period_index=11,
        axis_names=list(AXIS_NAMES),
        signed_steps=list(SIGNED_STEPS),
        previous_command=previous.copy(),
        input_min=grid["input_min"].copy(),
        input_max=grid["input_max"].copy(),
        anchor_commands=anchor.copy(),
        grid=grid,
        max_responses=max_responses,
        planned_request_count=14,
        response_evaluations=len(entries),
        unique_command_count=len({row["command_key"] for row in ledger}),
        total_iterations=0,
        request_ledger=ledger,
        evaluated_candidates=entries,
        terminal_row_index=historical_offset,
        terminal_values=terminal_values,
        prior_max_abs_residual=residual,
        safety_buffer=SAFETY_BUFFER,
        required_prefix=REQUIRED_PREFIX,
        reserve_factor=RESERVE_FACTOR,
        reserve_target=target,
        terminal_threshold=SAFETY_BUFFER + target,
        repeatability=repeat,
        historical_anchor_comparison=history,
        causal_checks=causal,
        clipping_checks=clipping,
        candidate_checks=candidates,
        eligible_ranking=ranking,
        selected_response_index=selected_index,
        selected=selected,
        selected_summary=summary,
        improvement_found=selected_index is not None,
        sufficient_reserve_restored=sufficient,
        causal_checks_all_passed=all(row["passed"] for row in causal),
        control_qualified=False,
    )
