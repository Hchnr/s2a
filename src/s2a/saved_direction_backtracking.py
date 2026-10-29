"""Prepare and score a fixed saved-direction grid without running a response."""

import numpy as np

from s2a.geometry import finite, pose_error, validate_transform
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints

ALPHAS = (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0)
CASE_ORDER = ("prefix", "early", "late")
SAFETY_NAMES = ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")


def _index(value, name, lower, upper):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or not lower <= value <= upper:
        raise ValueError(f"Integer {name} in [{lower}, {upper}] required")
    return int(value)


def _normalized(errors):
    return finite(
        np.stack(
            (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2),
            axis=1,
        )
    )


def build_batch(diagnostic, producers):
    """Keep every registered main pair and its six fixed interpolation points.

    The caller validates provenance, main-subset eligibility, common causal
    inputs, and the real batch's96/64/32 counts. This function never runs a
    response, clips a command, deduplicates endpoints, or adds adaptive points.
    Endpoints copy saved float64 bytes, including signed zeros; interior points
    use exactly ``base + alpha * (trial - base)``. Returned arrays own their data.
    """
    batch = []
    for name in CASE_ORDER:
        case = diagnostic["cases"][name]
        answer = producers[name]["answer"]
        horizon = _index(case["horizon"], "horizon", 1, 16)
        if answer["horizon"] != horizon:
            raise ValueError("Diagnostic and saved response horizons must match")
        target = _index(case["target_index"], "target_index", 0, horizon - 1)
        trace = answer["evaluated_candidates"]
        if not isinstance(trace, list) or not trace:
            raise ValueError("Nonempty saved response trace required")
        for pair_index, pair in enumerate(case["pairs"]):
            if not isinstance(pair["main_subset"], (bool, np.bool_)):
                raise ValueError("Boolean registered main-subset label required")
            if not pair["main_subset"]:
                continue
            base_index = _index(pair["base_index"], "base_index", 0, len(trace) - 1)
            trial_index = _index(pair["trial_index"], "trial_index", 0, len(trace) - 1)
            prefix = _index(pair["prefix_before"], "prefix_before", 0, horizon)
            base = finite(trace[base_index]["commands"], (horizon, 6))
            trial = finite(trace[trial_index]["commands"], (horizon, 6))
            normalized = _normalized(finite(trace[base_index]["tcp_errors"], (horizon, 6)))
            if not np.array_equal(normalized, finite(pair["qbase"], (horizon, 2))):
                raise ValueError("Registered qbase must equal the original saved TCP-error scores")
            base_target_max = float(np.max(normalized[target]))
            for alpha_index, alpha in enumerate(ALPHAS):
                if alpha_index == 0:
                    commands, source_index = base.copy(), base_index
                elif alpha_index == len(ALPHAS) - 1:
                    commands, source_index = trial.copy(), trial_index
                else:
                    commands = finite(base + alpha * (trial - base)).copy()
                    source_index = None
                batch.append(
                    dict(
                        case_name=name,
                        pair_index=pair_index,
                        base_index=base_index,
                        trial_index=trial_index,
                        alpha_index=alpha_index,
                        alpha=alpha,
                        prefix_before=prefix,
                        target_index=target,
                        horizon=horizon,
                        commands=commands,
                        source_trace_index=source_index,
                        base_target_max=base_target_max,
                    )
                )
    return batch


def score_backtracking_response(
    rollout,
    commands,
    previous,
    interface,
    lower,
    upper,
    *,
    past_inner_error_rad,
    terminal_normals,
    desired_tcp,
    site_from_tcp,
    prefix_before,
    target_index,
    base_target_max,
):
    """Score supplied predictions with the unchanged audit and pose thresholds.

    The caller owns all response/reset accounting. A complete model witness
    does not need the extra optimization buffer or a strict improvement; the
    separate feasible-descent label requires both. Neither label qualifies
    physical control or establishes an optimizer accepted step.
    """
    commands = finite(commands)
    if commands.ndim != 2 or commands.shape[1] != 6 or not 1 <= len(commands) <= 16:
        raise ValueError("One to sixteen six-axis commands required")
    horizon = len(commands)
    prefix_before = _index(prefix_before, "prefix_before", 0, horizon)
    target_index = _index(target_index, "target_index", 0, horizon - 1)
    base_target_max = float(finite(base_target_max, ()))
    if base_target_max < 0:
        raise ValueError("Nonnegative base target squared error required")
    poses = validate_transform(rollout["pose"])
    desired = validate_transform(desired_tcp)
    tool = validate_transform(site_from_tcp)
    if poses.shape != (horizon, 4, 4) or desired.shape != (horizon, 4, 4) or tool.shape != (4, 4):
        raise ValueError("Horizon-matched SITE/desired poses and one SITE-from-TCP transform required")
    audit = audit_joint_preview_constraints(
        rollout,
        commands,
        previous,
        interface,
        lower,
        upper,
        past_inner_error_rad=past_inner_error_rad,
        terminal_normals=terminal_normals,
        braking_time_s=0.1,
    )
    errors = pose_error(desired, poses @ tool)
    normalized = _normalized(errors)
    failures = np.flatnonzero(np.any(normalized > 1.0, axis=1))
    prefix_length = int(failures[0]) if len(failures) else horizon
    safety = np.concatenate([audit["groups"][name]["slack"].ravel() for name in SAFETY_NAMES])
    safety_min = float(np.min(safety))
    joint_feasible = bool(audit["feasible"])
    prefix_preserved = prefix_length >= prefix_before
    buffer_satisfied = safety_min >= 1e-8
    target_max = float(np.max(normalized[target_index]))
    improvement = base_target_max - target_max
    target_improved = improvement > 1e-12
    return dict(
        audit=audit,
        tcp_errors=errors,
        normalized_squared_errors=normalized,
        safety_min=safety_min,
        prefix_length=prefix_length,
        joint_feasible=joint_feasible,
        prefix_preserved=prefix_preserved,
        buffer_satisfied=buffer_satisfied,
        target_max=target_max,
        target_improvement=improvement,
        target_improved=target_improved,
        target_passed=bool(np.all(normalized[target_index] <= 1.0)),
        feasible=bool(joint_feasible and prefix_length == horizon),
        feasible_descent=bool(joint_feasible and prefix_preserved and buffer_satisfied and target_improved),
    )
