"""Audit pose-restricted dual recovery without executing NNLS or a QP.

The frozen GN audit retains the full original ledger and KKT gates. Saved
returned points determine recovery subsets; callback execution remains the
separately fingerprinted core QA's responsibility.
"""

import numpy as np
from directional_derivative_audit import array, exact_array, integer, numerical_compare, require
from state_vector_gn_search_audit import _causal_update, _error, _periods, _pose, _problem
from state_vector_gn_search_audit import audit_search as audit_gn_search
from state_vector_gn_search_audit import reconstruct_error_vector_jacobian as reconstruct_error_vector_jacobian
from state_vector_objective_scale_audit import _divisor

QP_OBJECTIVE_SCALING_POLICY = "pose_whole_objective_divisor"
RESTORE_DUAL_PROVENANCE = "original_nnls_certificate_divided_by_objective_divisor"
POSE_DUAL_PROVENANCE = "unscaled_gradient_subset_nnls_certificate_divided_by_objective_divisor"
DUAL_RECOVERY_POLICY = "pose_slack_restricted_nnls_restore_full_active"


def _scale_record(qp, B, g, A, b, divisor, mode, compare):
    """Bind algebraic metadata in addition to the original-unit certificate."""
    require(
        qp["objective_scaled_multipliers_provenance"]
        == (POSE_DUAL_PROVENANCE if mode == "pose" else RESTORE_DUAL_PROVENANCE),
        "Algebraic original-NNLS dual provenance",
    )
    saved_divisor = qp["solver_objective_divisor"]
    require(type(saved_divisor) is float, "Explicit floating-point whole-objective divisor")
    require(np.isfinite(saved_divisor) and saved_divisor >= 1.0, "Finite positive objective divisor")
    compare("Independently reconstructed objective divisor", divisor, saved_divisor)
    original_value, mu = qp["objective"], qp["scaled_multipliers"]
    if original_value is None:
        require(qp["solver_objective_value"] is None, "No fabricated incomplete objective")
    else:
        point = array(qp["point"], "Original QP point", (73,))
        d = point[:-1]
        value = float(point[-1] + g @ d + 0.5 * (d @ (B @ d)))
        compare("Original objective before scaling", value, original_value)
        compare(
            "Entire objective including t divided by C", original_value / saved_divisor, qp["solver_objective_value"]
        )
    maximum_roundtrip = None
    if mu is None:
        require(qp["objective_scaled_multipliers"] is None, "No fabricated incomplete dual")
    else:
        mu = array(mu, "Original canonical NNLS multipliers", (len(A),))
        nu = array(qp["objective_scaled_multipliers"], "Algebraic objective-scaled dual", (len(A),))
        require(np.all(nu >= 0), "Positive scaling preserves nonnegative multipliers")
        exact_array("Algebraic dual is original NNLS mu/C", mu / saved_divisor, nu)
        compare("Inverse dual conversion to original canonical units", mu, saved_divisor * nu)
        scale = np.maximum.reduce((np.ones(len(A)), np.max(np.abs(A), axis=1), np.abs(b)))
        compare("Inverse dual conversion to original raw units", qp["multipliers"], saved_divisor * nu / scale)
        maximum_roundtrip = float(np.max(np.abs(saved_divisor * nu - mu)))
    return dict(
        divisor=saved_divisor,
        independent_divisor_estimate=divisor,
        original_objective=original_value,
        solver_objective_value=qp["solver_objective_value"],
        solver_success=qp["solver_success"],
        original_certificate_qualified=qp["qualified"],
        maximum_canonical_dual_roundtrip_difference=maximum_roundtrip,
    )


def _recovery_record(qp, B, g, A, b, mode, compare):
    """Rebuild both sets from the returned original point and count saved calls."""
    require(mode in ("restore", "pose") and qp["dual_recovery_mode"] == mode, "Exact stage dual-recovery mode")
    tolerance = qp["certificate_tolerance"]
    full_threshold = 10.0 * tolerance
    recovery_threshold = (0.01 if mode == "pose" else 10.0) * tolerance
    for key, expected in (("full_active_threshold", full_threshold), ("recovery_threshold", recovery_threshold)):
        require(type(qp[key]) is float and np.isfinite(qp[key]), "Finite explicit recovery threshold: " + key)
        exact_array("Original threshold expression: " + key, expected, qp[key])
    complete_point = qp["point"] is not None and qp["objective"] is not None
    if complete_point:
        point = array(qp["point"], "Returned original QP point", (73,))
        scale = np.maximum.reduce((np.ones(len(A)), np.max(np.abs(A), axis=1), np.abs(b)))
        slack = (A / scale[:, None]) @ point + b / scale
        full = np.flatnonzero(slack <= full_threshold).tolist()
        recovery = [index for index in full if slack[index] <= recovery_threshold]
        require(qp["nnls_max_iterations"] == 3 * max(len(full), 1), "Original full-active NNLS iteration budget")
    else:
        full, recovery = [], []
    for key, expected in (("active_indices", full), ("recovery_indices", recovery)):
        require(
            isinstance(qp[key], list) and all(type(index) is int for index in qp[key]) and qp[key] == expected,
            "Independently reconstructed complete indices: " + key,
        )
    attempts = int(complete_point and bool(recovery))
    integer(qp["nnls_fit_attempts"], attempts, "Exactly one NNLS call for a nonempty available subset")
    if qp["scaled_multipliers"] is not None:
        mu = array(qp["scaled_multipliers"], "Complete original canonical multipliers", (len(A),))
        excluded = np.ones(len(A), dtype=bool)
        excluded[recovery] = False
        require(np.all(mu[excluded] == 0.0), "All rows outside the recovery subset have exact zero mu")
        if not recovery:
            require(
                complete_point and qp["nnls_success"] is True and qp["nnls_error"] is None,
                "Completed empty algebraic recovery",
            )
            gradient = np.r_[g + B @ point[:72], 1.0]
            compare(
                "Empty subset residual is original returned-point gradient norm",
                np.linalg.norm(gradient),
                qp["nnls_residual_norm"],
            )
    elif complete_point and attempts:
        require(qp["nnls_success"] is False, "No complete dual after failed NNLS or incomplete certificate")
    successful = int(attempts == 1 and qp["nnls_success"] is True)
    return dict(
        mode=mode,
        full_active_threshold=full_threshold,
        recovery_threshold=recovery_threshold,
        full_active_count=len(full),
        recovery_count=len(recovery),
        full_active_indices=full,
        recovery_indices=recovery,
        nnls_max_iterations=qp["nnls_max_iterations"],
        nnls_fit_attempts=attempts,
        successful_nnls_fits=successful,
        failed_nnls_fits=attempts - successful,
        original_certificate_qualified=qp["qualified"],
    )


def audit_search(
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
):
    """Retain every original gate, then bind the additional solver-scale record."""
    checked = audit_gn_search(
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
        B, g, A, b = _problem(
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
        scope="Original full GN/KKT, returned-point recovery sets, declared call counts and scaling algebra; actual callback execution requires separately bound core QA",
    )
