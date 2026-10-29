"""Finite tail-command repair with full trajectory and causal-prefix checks.

The terminal slack is an observed numerical-model response, not an empirical
error bound. A successful repair remains separate from full-H12 pose feasibility
and cannot establish physical control or task qualification.
"""

from copy import deepcopy

import numpy as np

from s2a.directional_derivative_probe import _canonical, _repeat_comparison
from s2a.geometry import finite, pose_error, validate_transform
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints
from s2a.tail_translation_reserve import (
    AXIS_NAMES,
    REQUIRED_PREFIX,
    SAFETY_BUFFER,
    SAFETY_NAMES,
    SIGNED_STEPS,
    _causal_checks,
    build_tail_translation_grid,
)
from s2a.terminal_secant_probe import _clipping_checks, _terminal

ENTRY_FIELDS = (
    "commands",
    "rollout",
    "audit",
    "tcp_errors",
    "normalized_squared_errors",
    "safety_slack",
    "prefix_length",
    "maximum_normalized_squared_error",
    "maximum_pose_cap_violation",
    "pose_loss",
    "feasible",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _select(entries, ledger, terminals, causal, historical, repeat):
    candidates = []
    for index in range(1, 13):
        entry, request = entries[index], ledger[index]
        safety = entry["safety_slack"]
        gates = dict(
            joint_audit=entry["audit"]["feasible"],
            safety_buffer=bool(np.all(safety >= SAFETY_BUFFER)),
            prefix9=entry["prefix_length"] >= REQUIRED_PREFIX,
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
                terminal=terminals[index],
                terminal_improvement=terminals[index] - terminals[0],
                other_safety_min=float(np.min(safety[:-1])),
                safety_min=float(np.min(safety)),
                joint_feasible=gates["joint_audit"],
                safety_buffer_satisfied=gates["safety_buffer"],
                prefix9_preserved=gates["prefix9"],
                causal_exact=causal[index]["passed"],
                historical_exact=gates["historical_anchor"],
                repeat_exact=gates["anchor_repeat"],
                repair_eligible=all(gates.values()),
                rejection_reasons=[key for key, passed in gates.items() if not passed],
            )
        )
    eligible = sorted(
        (row for row in candidates if row["repair_eligible"]),
        key=lambda row: (-row["terminal"], row["actual_displacement_linf"], row["request_index"]),
    )
    ranking = [row["response_index"] for row in eligible]
    best = eligible[0] if eligible and eligible[0]["terminal"] > terminals[0] else None
    if best is None:
        return candidates, ranking, None, None, None
    index = best["response_index"]
    selected = deepcopy(entries[index])
    summary = dict(
        **deepcopy(best),
        prefix_length=selected["prefix_length"],
        tcp_errors=selected["tcp_errors"].copy(),
        normalized_squared_errors=selected["normalized_squared_errors"].copy(),
        feasible=selected["feasible"],
    )
    return candidates, ranking, index, selected, summary


def probe_terminal_only_tail_repair(
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
    prior_max_abs_residual=None,
):
    """Charge anchor, twelve fixed translations, and an exact anchor repeat.

    All thirteen distinct commands must pass before the first call. Each call
    returns a complete H12 response from the same initial state. Full prefix
    records and both goal-rotation representations must remain exact for any
    selected candidate; historical and repeated anchor agreement are global
    selection gates. There is no inferred empirical uncertainty threshold.
    """
    _require(prior_max_abs_residual is None, "This repair does not accept an empirical error bound")
    interface = deepcopy(interface)
    rebuilt = build_tail_translation_grid(anchor, previous, interface.input_min, interface.input_max)
    _require(_canonical(rebuilt) == _canonical(grid), "Complete fixed grid must match input generation")
    grid = rebuilt
    _require(grid["admitted"], "Anchor and all twelve distinct nonzero commands must pass preflight")
    anchor, previous = grid["anchor_commands"], grid["previous_command"]
    lower, upper = finite(lower, (7,)).copy(), finite(upper, (7,)).copy()
    _require(np.all(lower < upper), "Original seven-joint bounds")
    desired, tool = validate_transform(desired).copy(), validate_transform(site_from_tcp).copy()
    _require(
        desired.shape == (12, 4, 4) and tool.shape == (4, 4),
        "All twelve original reference endpoints and TCP transform",
    )
    profile, normals = finite(past_inner_error_rad, (25, 7)).copy(), finite(terminal_normals, (1, 7)).copy()
    _require(
        np.all(profile >= 0) and np.all(np.diff(profile, axis=0) >= 0),
        "Original nonnegative nondecreasing history profile",
    )
    _require(np.array_equal(normals[0], np.eye(7)[6]), "Original e7 terminal row")
    _require(isinstance(historical_anchor, dict), "Complete historical entry required")
    _require(set(ENTRY_FIELDS) <= set(historical_anchor), "Every generated numeric entry field is required in history")
    historical = {key: deepcopy(historical_anchor[key]) for key in ENTRY_FIELDS}
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
        return dict(
            commands=commands.copy(),
            rollout=rollout,
            audit=audit,
            tcp_errors=errors,
            normalized_squared_errors=normalized,
            safety_slack=np.concatenate([audit["groups"][name]["slack"].ravel() for name in SAFETY_NAMES]),
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
        _require(index < 14, "No extra response outside the fixed grid")
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
    terminals = []
    for entry in entries:
        offset, terminal = _terminal(entry, normals, 12, 7)
        _require(offset == historical_offset, "Consistent complete safety-vector terminal row")
        terminals.append(terminal)
    repeat = _repeat_comparison(entries[0], entries[-1], 13)
    comparison = _repeat_comparison(historical, entries[0], 0)
    del comparison["initial_response_index"], comparison["repeated_response_index"]
    history = dict(
        new_response_index=0,
        compared_fields=list(ENTRY_FIELDS),
        ignored_historical_fields=sorted(set(historical_anchor) - set(ENTRY_FIELDS)),
        comparison=comparison,
    )
    causal, clipping = _causal_checks(entries), _clipping_checks(entries, 12, 7)
    candidates, ranking, selected_index, selected, summary = _select(
        entries, ledger, terminals, causal, history, repeat
    )
    return dict(
        method="fixed_terminal_only_tail_repair",
        horizon=12,
        tail_period_index=11,
        axis_names=list(AXIS_NAMES),
        signed_steps=list(SIGNED_STEPS),
        previous_command=previous.copy(),
        input_min=grid["input_min"].copy(),
        input_max=grid["input_max"].copy(),
        anchor_commands=anchor.copy(),
        grid=grid,
        max_responses=14,
        planned_request_count=14,
        response_evaluations=len(entries),
        unique_command_count=len({row["command_key"] for row in ledger}),
        total_iterations=0,
        request_ledger=ledger,
        evaluated_candidates=entries,
        terminal_row_index=historical_offset,
        terminal_values=terminals,
        safety_buffer=SAFETY_BUFFER,
        required_prefix=REQUIRED_PREFIX,
        empirical_error_bound_used=False,
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
        causal_checks_all_passed=all(row["passed"] for row in causal),
        control_qualified=False,
    )
