"""Prepare and audit a fixed pair of reference schedules without running a response."""

from copy import deepcopy

import numpy as np

from s2a.geometry import finite, pose_error, validate_transform
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints
from s2a.reference_governor import interpolate_pose


def _command_bounds(commands, previous, input_min, input_max):
    lower = np.tile(input_min, (len(commands), 1))
    upper = np.tile(input_max, (len(commands), 1))
    lower[0] = np.maximum(lower[0], previous - 0.25)
    upper[0] = np.minimum(upper[0], previous + 0.25)
    if (
        np.any(input_min >= input_max)
        or np.any(upper - lower <= 1e-4)
        or np.any(commands < lower - 1e-12)
        or np.any(commands > upper + 1e-12)
        or np.any(np.abs(np.diff(np.vstack((previous, commands)), axis=0)) > 0.25 + 1e-12)
    ):
        raise ValueError("Warm commands require the original box/rate bounds and nondegenerate difference intervals")
    return lower, upper


def prepare_paired_reference_phases(targets_tcp, openings, warm12, *, previous, interface, case_name=None):
    """Snapshot the fixed early/late H13 schedules from a generated reference.

    The only new command is the mean of the neighboring original commands.
    All original nodes 0..11 remain; original timed scoring also needs node 12.
    This function checks commands, not dynamic safety. It never predicts,
    searches, restores safety, consumes a reference, or reads later states.
    ``case_name`` isolates command admission for one independently run case.
    """
    targets = validate_transform(targets_tcp).copy()
    if targets.ndim != 3 or len(targets) < 13:
        raise ValueError("At least thirteen generated TCP endpoints are required")
    openings = finite(openings, (len(targets),)).copy()
    if np.any((openings < 0) | (openings > 1)):
        raise ValueError("Generated openings must lie in [0,1]")
    source_warm = finite(warm12, (12, 6)).copy()
    previous = finite(previous, (6,)).copy()
    input_min = finite(interface.input_min, (6,)).copy()
    input_max = finite(interface.input_max, (6,)).copy()
    _command_bounds(source_warm, previous, input_min, input_max)
    if case_name is not None and case_name not in ("early", "late"):
        raise ValueError("case_name must be early, late, or None")

    result = {}
    for name, left in (("early", 5), ("late", 8)):
        if case_name is not None and name != case_name:
            continue
        insertion = left + 1
        phases = np.insert(np.arange(12, dtype=float), insertion, left + 0.5)
        desired = np.insert(targets[:12], insertion, interpolate_pose(targets[left], targets[left + 1], 0.5), axis=0)
        opening = np.insert(openings[:12], insertion, (openings[left] + openings[left + 1]) / 2)
        warm = np.insert(source_warm, insertion, (source_warm[left] + source_warm[left + 1]) / 2, axis=0)
        lower, upper = _command_bounds(warm, previous, input_min, input_max)
        used_start = np.clip(warm, lower, upper)
        if used_start.tobytes() != warm.tobytes():
            raise ValueError(f"{name} warm requires roundoff clipping; the registered command bytes must be unchanged")
        nodes = np.arange(12)
        result[name] = dict(
            name=name,
            phases=phases,
            desired_tcp=desired,
            opening=opening,
            original_timed_tcp=targets[:13].copy(),
            original_timed_opening=openings[:13].copy(),
            original_timed_cursors=np.arange(13),
            original_node_cursors=nodes.copy(),
            original_node_endpoint_indices=nodes + (nodes >= insertion),
            unconsumed_suffix=np.arange(12, len(targets)),
            warm=warm,
            source_warm12=source_warm.copy(),
            used_start=used_start,
            start_roundoff_clip=used_start - warm,
            physical_time_s=0.05 * np.arange(1, 14),
            reference_time_s=0.05 * (phases + 1),
            duration_s=0.65,
            control_dt_s=0.05,
            scheduled_original_node_count=12,
            scheduled_final_original_cursor=11,
            actual_consumed_reference_count=0,
        )
    return result


def _pose_metrics(desired, measured):
    errors = pose_error(desired, measured)
    normalized = finite(
        np.stack(
            (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2),
            axis=1,
        )
    )
    passed = np.all(normalized <= 1.0, axis=1)
    failures = np.flatnonzero(~passed)
    return dict(
        tcp_errors=errors,
        normalized_squared_errors=normalized,
        pose_cap_pass=passed,
        all_pose_caps_satisfied=bool(np.all(passed)),
        prefix_length=int(failures[0]) if len(failures) else len(passed),
        maximum_normalized_squared_error=float(np.max(normalized)),
        maximum_pose_cap_violation=float(max(0.0, np.max(np.sqrt(normalized)) - 1.0)),
        pose_loss=float(np.sum(normalized)),
    )


def score_phase_response(site_poses, case, site_from_tcp):
    """Keep nominal-time, retimed-path and original-node TCP metrics distinct.

    No opening error is inferred from the fixed-finger arm response. These are
    predicted pose metrics, not joint qualification or actual path consumption.
    SITE endpoints are transformed once into the same registered TCP frame.
    """
    poses = validate_transform(site_poses)
    tool = validate_transform(site_from_tcp)
    desired = validate_transform(case["desired_tcp"])
    timed = validate_transform(case["original_timed_tcp"])
    indices = np.asarray(case["original_node_endpoint_indices"])
    if (
        poses.shape != (13, 4, 4)
        or tool.shape != (4, 4)
        or desired.shape != (13, 4, 4)
        or timed.shape != (13, 4, 4)
        or indices.shape != (12,)
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= 13)
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError("Thirteen SITE/target endpoints and twelve ordered original-node indices required")
    predicted_tcp = poses @ tool
    original_nodes = _pose_metrics(timed[:12], predicted_tcp[indices])
    return dict(
        predicted_tcp=predicted_tcp,
        original_timed=_pose_metrics(timed, predicted_tcp),
        retimed_path=_pose_metrics(desired, predicted_tcp),
        original_nodes=original_nodes,
        predicted_qualified_original_node_prefix=original_nodes["prefix_length"],
        predicted_final_qualified_original_cursor=original_nodes["prefix_length"] - 1,
        actual_consumed_reference_count=0,
    )


def assess_phase_warm(
    rollout,
    case,
    previous,
    interface,
    lower,
    upper,
    site_from_tcp,
    *,
    past_inner_error_rad,
    terminal_normals,
):
    """Audit one caller-supplied complete warm response without running a search.

    The returned initial record has the frozen prefix solver's entry schema.
    A complete witness may bypass its optimization buffer; an incomplete warm
    must pass both the full audit and 1e-8 nonlinear buffer. Admission failure
    is distinct from a finite search without a witness. The caller owns actual
    response accounting and must charge this initial response exactly once.
    """
    commands = finite(case["warm"], (13, 6)).copy()
    lower_command, upper_command = _command_bounds(
        commands, finite(previous, (6,)), finite(interface.input_min, (6,)), finite(interface.input_max, (6,))
    )
    if np.clip(commands, lower_command, upper_command).tobytes() != commands.tobytes():
        raise ValueError("Warm command bytes must equal the frozen solver's unclipped start")
    lower = finite(lower)
    normals = finite(terminal_normals)
    if lower.ndim != 1 or not len(lower) or normals.ndim != 2 or normals.shape[1] != len(lower):
        raise ValueError("Nonempty joint limits and two-dimensional terminal direction rows required")
    snapshot = deepcopy(rollout)
    audit = audit_joint_preview_constraints(
        snapshot,
        commands,
        previous,
        interface,
        lower,
        upper,
        past_inner_error_rad=past_inner_error_rad,
        terminal_normals=normals,
        braking_time_s=0.1,
    )
    metrics = score_phase_response(snapshot["pose"], case, site_from_tcp)
    retimed = metrics["retimed_path"]
    safety = np.concatenate(
        [
            audit["groups"][name]["slack"].ravel()
            for name in ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")
        ]
    )
    complete = bool(audit["feasible"] and retimed["all_pose_caps_satisfied"])
    safety_min = float(np.min(safety))
    admitted = bool(complete or (audit["feasible"] and safety_min >= 1e-8))
    reason = None
    if not admitted:
        reason = "joint_or_command_audit_failed" if not audit["feasible"] else "optimization_safety_buffer_not_met"
    initial = dict(
        commands=commands,
        rollout=snapshot,
        audit=audit,
        **{
            key: deepcopy(retimed[key])
            for key in (
                "tcp_errors",
                "normalized_squared_errors",
                "prefix_length",
                "maximum_normalized_squared_error",
                "maximum_pose_cap_violation",
                "pose_loss",
            )
        },
        safety_slack=safety,
        feasible=complete,
    )
    return dict(
        initial=initial,
        metrics=metrics,
        joint_feasible=audit["feasible"],
        safety_min=safety_min,
        complete_witness=complete,
        admitted=admitted,
        status="complete_witness" if complete else "admitted" if admitted else "warm_admission_failed",
        rejection_reason=reason,
    )
