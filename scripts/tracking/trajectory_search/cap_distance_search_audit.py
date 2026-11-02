"""Audit the cap-distance ledger and unchanged original-unit dual certificates."""

import numpy as np

from scripts.tracking.trajectory_search.cap_distance_audit_policy import CapDistanceAuditPolicy
from scripts.tracking.trajectory_search.objective_search_audit import audit_search as audit_objective_search
from scripts.tracking.trajectory_search.state_vector_dual_recovery_audit import (
    DUAL_RECOVERY_POLICY,
    QP_OBJECTIVE_SCALING_POLICY,
    _recovery_record,
    _scale_record,
)
from scripts.tracking.trajectory_search.state_vector_gn_search_audit import _causal_update, _error, _periods, _pose
from scripts.tracking.trajectory_search.state_vector_objective_scale_audit import _divisor
from scripts.verification.trajectory_search.directional_derivative_audit import array, numerical_compare, require


def audit_cap_search(
    answer,
    initial_Js,
    initial_Jq,
    initial_Je,
    historical_initial,
    previous,
    input_min,
    input_max,
    config,
    compare=numerical_compare,
    *,
    domain_reference=None,
    preserve_pose_reserve=False,
):
    """Retain every original gate, then bind the additional solver-scale record."""
    checked = audit_objective_search(
        answer,
        initial_Js,
        initial_Jq,
        initial_Je,
        historical_initial,
        previous,
        input_min,
        input_max,
        config,
        compare=compare,
        objective=CapDistanceAuditPolicy(),
        domain_reference=domain_reference,
        preserve_pose_reserve=preserve_pose_reserve,
    )
    require(answer["qp_objective_scaling_policy"] == QP_OBJECTIVE_SCALING_POLICY, "Whole-objective scaling policy")
    require(answer["dual_recovery_policy"] == DUAL_RECOVERY_POLICY, "Pose-only restricted recovery policy")
    Js, Jq, Je = (
        array(initial_Js, "Initial original safety model", (4369, 72)).copy(),
        array(initial_Jq, "Initial original q model", (24, 72)).copy(),
        array(initial_Je, "Initial original residual model", (72, 72)).copy(),
    )
    safety_periods, pose_periods = _periods()
    error_periods = np.repeat(np.arange(12), 6)
    entries, scale_checks, recovery_checks = answer["evaluated_candidates"], [], []
    for stage in answer["stages"]:
        anchor = entries[stage["anchor_before_index"]]
        x, safety, squared, error = (
            array(anchor["commands"], "Actual numerical anchor command", (12, 6)),
            array(anchor["safety_slack"], "Original anchor safety", (4369,)),
            _pose(anchor),
            _error(anchor),
        )
        B, g, A, b = CapDistanceAuditPolicy.problem(
            x,
            safety,
            squared,
            error,
            Js,
            Jq,
            Je,
            np.asarray(previous),
            np.asarray(answer["global_lower"]),
            np.asarray(answer["global_upper"]),
            np.asarray(answer["pose_ceiling"]),
            stage["radius_before"],
            stage["mode"],
            config,
        )
        divisor = _divisor(B, g, stage["radius_before"], stage["mode"])
        record = _scale_record(stage["qp"], B, g, A, b, divisor, stage["mode"], compare)
        scale_checks.append(dict(stage_index=stage["stage_index"], mode=stage["mode"], **record))
        recovery_checks.append(
            dict(stage_index=stage["stage_index"], **_recovery_record(stage["qp"], B, g, A, b, stage["mode"], compare))
        )
        source = stage["accepted_response_index"]
        if source is None and stage["rejected_update"] is not None:
            source = stage["rejected_update"]["source_response_index"]
        if source is not None:
            observed = entries[source]
            step = (np.asarray(observed["commands"]) - x).ravel()
            Js, _ = _causal_update(Js, step, np.asarray(observed["safety_slack"]) - safety, safety_periods, config)
            Jq, _ = _causal_update(Jq, step, _pose(observed) - squared, pose_periods, config)
            Je, _ = _causal_update(Je, step, _error(observed) - error, error_periods, config)
    return dict(
        **checked,
        qp_objective_scaling_policy=QP_OBJECTIVE_SCALING_POLICY,
        objective_scaling_checks=scale_checks,
        dual_recovery_policy=DUAL_RECOVERY_POLICY,
        dual_recovery_checks=recovery_checks,
        total_nnls_fit_attempts=sum(row["nnls_fit_attempts"] for row in recovery_checks),
        total_successful_nnls_fits=sum(row["successful_nnls_fits"] for row in recovery_checks),
        total_failed_nnls_fits=sum(row["failed_nnls_fits"] for row in recovery_checks),
        solver_callback_execution_verified_here=False,
        nnls_callback_execution_verified_here=False,
        scaled_dual_scope="Algebraic unscaled-gradient recovery mu/C, with mode-specific provenance; not returned by SLSQP",
        scope="Independent cap projection/GN/KKT, returned-point recovery sets and scaling algebra; actual callback execution requires separately bound core QA",
    )
