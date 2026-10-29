"""Audit original GN/KKT records plus a declared whole-objective scale.

The frozen GN audit owns the complete unchanged state machine. This module
independently rebuilds the chronological local models and scale metadata. It
does not execute or observe solver callbacks, a solver, or a dynamics evaluator.
"""

import numpy as np
from directional_derivative_audit import array, exact_array, numerical_compare, require
from state_vector_gn_search_audit import _causal_update, _error, _periods, _pose, _problem
from state_vector_gn_search_audit import audit_search as audit_gn_search
from state_vector_gn_search_audit import reconstruct_error_vector_jacobian as reconstruct_error_vector_jacobian

QP_OBJECTIVE_SCALING_POLICY = "pose_whole_objective_divisor"
SCALED_DUAL_PROVENANCE = "original_nnls_certificate_divided_by_objective_divisor"


def _divisor(B, g, radius, mode):
    if mode == "restore":
        return 1.0
    require(mode == "pose", "Registered GN or unchanged restoration mode")
    return max(1.0, float(np.max(np.abs(g))), float(radius) * float(np.linalg.norm(B, ord=np.inf)))


def _scale_record(qp, B, g, A, b, divisor, compare):
    """Bind algebraic metadata in addition to the original-unit certificate."""
    require(
        qp["objective_scaled_multipliers_provenance"] == SCALED_DUAL_PROVENANCE,
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
    Js, Jq, Je = (
        array(initial_Js, "Initial original safety model", (4369, 72)).copy(),
        array(initial_Jq, "Initial original q model", (24, 72)).copy(),
        array(initial_Je, "Initial original residual model", (72, 72)).copy(),
    )
    safety_periods, pose_periods = _periods()
    error_periods = np.repeat(np.arange(12), 6)
    entries, scale_checks = answer["evaluated_candidates"], []
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
        record = _scale_record(stage["qp"], B, g, A, b, divisor, compare)
        scale_checks.append(dict(stage_index=stage["stage_index"], mode=stage["mode"], **record))
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
        solver_callback_execution_verified_here=False,
        scaled_dual_scope="Algebraic original-NNLS mu/C; SLSQP does not return these duals",
        scope="Original GN ledger/KKT and declared scaling algebra; actual callbacks require separately bound core QA",
    )
