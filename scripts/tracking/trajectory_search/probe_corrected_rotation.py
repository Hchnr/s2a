"""One independently audited rotation QP after a saved causal safety update."""

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.research.trajectory_search.interior_trajectory_search import causal_periods, causal_secant_update
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, read, reference, require
from s2a.research.trajectory_search.state_vector_dual_recovery import objective_divisor, solve_recovered_epigraph_qp
from scripts.tracking.trajectory_search.cap_distance_audit_policy import CapDistanceAuditPolicy, projection_model
from scripts.tracking.trajectory_search.probe_rotation_constraints import (
    preflight,
    reconstruct_models,
    rotation_problem,
)
from scripts.tracking.trajectory_search.response_probe_runtime import ResponseProbeRuntime
from scripts.tracking.trajectory_search.state_vector_dual_recovery_audit import _recovery_record, _scale_record
from scripts.tracking.trajectory_search.state_vector_gn_search_audit import _causal_update, _error, _pose
from scripts.verification.trajectory_search.accepted_point_sqp_math_audit import audit_qp_certificate
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare

OUTPUT = REPO / "outputs/corrected_rotation_probe_20261102"
CONFIG = REPO / "configs/tracking/trajectory_search/corrected_rotation_probe_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/corrected_rotation_probe_20261102.md"
QA = REPO / "outputs/qa/corrected_rotation_probe_20261102/result.json"


def correct_safety(J, anchor, fit, config):
    """Fit only the explicitly supplied safety observation; preserve causal rows."""
    x, y = (np.asarray(e["commands"], dtype=float) for e in (anchor, fit))
    require(x.shape == y.shape == (12, 6), "Complete fixed H12 observations")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Finite commands")
    require(
        fit["audit"]["feasible"] and np.all(np.asarray(fit["safety_slack"]) >= config["safety_buffer"]),
        "Safe fit sample",
    )
    step = (y - x).ravel()
    require(0 < np.max(abs(step)) <= 1e-4 + 1e-12, "Only the fixed local fit scale")
    change = np.asarray(fit["safety_slack"]) - np.asarray(anchor["safety_slack"])
    periods, _ = causal_periods()
    updated, record = causal_secant_update(J, step, change, periods, config)
    independent, _ = _causal_update(J, step, change, periods, config)
    np.testing.assert_array_equal(updated, independent)
    return updated, record


def main():
    spec = read(reference(CONFIG))
    require(
        spec
        == dict(
            seed=42,
            fit_index=5,
            radius=0.005,
            qp_budget=1,
            qp_iteration_budget=100,
            nnls_fit_budget=1,
            response_count_per_side=3,
            total_query_budget=1874,
            total_inner_step_budget=1800,
            model_budget=2,
            timeout_seconds=600,
            automatic_retry=False,
        ),
        "Fixed corrected probe budget and sample",
    )
    runtime = ResponseProbeRuntime(OUTPUT, QA)
    local_ref = reference(REPO / "outputs/rotation_direction_locality_20261102/result.json")
    local = runtime.load(local_ref, "2aae6543c751cd9af7b6e43ac306c729495278d4725b94fd254fd9c5052b0242")
    require(local["passed"] and local["complete"] and local["sources_unchanged"], "Closed locality evidence")
    registration = runtime.load(local["registration_ref"])
    prior = runtime.load(registration["detail"]["prior_probe_ref"])
    old = runtime.load(prior["registration_ref"])
    cap = runtime.load(runtime.load(old["cap_result_ref"])["producer_final_ref"])["answer"]
    stencil = runtime.load(old["stencil_ref"])
    anchor = runtime.load(reference(REPO / "outputs/rotation_direction_locality_20261102/production/response_00.json"))
    fit = runtime.load(reference(REPO / "outputs/rotation_direction_locality_20261102/production/response_05.json"))
    models = reconstruct_models(cap, stencil, 58)
    np.testing.assert_array_equal(models["Js"], np.asarray(old["A"])[:4369, :72])
    models["Js"], update = correct_safety(models["Js"], anchor, fit, cap["config"])
    previous = np.asarray(runtime.source["previous_command"])[:6]
    lower, upper = (np.asarray(cap[k]) for k in ("global_lower", "global_upper"))
    ceiling = np.asarray(cap["pose_ceiling"])
    problem = rotation_problem(anchor, models, previous, lower, upper, ceiling, spec["radius"], cap["config"])
    _, _, A, b = CapDistanceAuditPolicy.problem(
        np.asarray(anchor["commands"]),
        np.asarray(anchor["safety_slack"]),
        _pose(anchor),
        _error(anchor),
        models["Js"],
        models["Jq"],
        models["Je"],
        previous,
        lower,
        upper,
        ceiling,
        spec["radius"],
        "pose",
        cap["config"],
    )
    r, J = projection_model(_error(anchor), _pose(anchor), models["Je"])
    B, g = np.eye(72) + J[-3:].T @ J[-3:], J[-3:].T @ r[-3:]
    for name, value in dict(B=B, g=g, A=A, b=b).items():
        np.testing.assert_array_equal(problem[name], value)
    prereg = publish_result(
        OUTPUT / "qp_registration.json",
        dict(
            spec=spec,
            config_ref=reference(CONFIG),
            plan_ref=reference(PLAN),
            qa_ref=runtime.qa_ref,
            input_refs=list(runtime.refs.values()),
            B=B,
            g=g,
            A=A,
            b=b,
            initial_t=problem["initial_t"],
            safety_update=update,
            new_causal_secant_updates=1,
            independent_update_reconstructions=1,
        ),
    )
    divisor = objective_divisor(problem, spec["radius"], "pose")
    qp = solve_recovered_epigraph_qp(
        B,
        g,
        A,
        b,
        problem["initial_t"],
        objective_divisor=divisor,
        dual_recovery_mode="pose",
        max_iterations=100,
        certificate_tolerance=1e-6,
    )

    def compare(n, a, b):
        return semantic_compare(n, a, b, 1e-10)

    certificate = audit_qp_certificate(qp, B, g, A, b)
    scale = _scale_record(qp, B, g, A, b, divisor, "pose", compare)
    recovery = _recovery_record(qp, B, g, A, b, "pose", compare)
    receipt = publish_result(
        OUTPUT / "qp_result.json",
        dict(
            registration_ref=prereg,
            qp=qp,
            certificate=certificate,
            scale=scale,
            recovery=recovery,
            qp_solves=1,
            nnls_fit_attempts=qp["nnls_fit_attempts"],
            control_qualified=False,
        ),
    )
    require(qp["qualified"] and qp["nnls_fit_attempts"] <= 1, "One qualified QP within registered budget")
    x = np.asarray(anchor["commands"])
    command = x + np.asarray(qp["point"])[:72].reshape(12, 6)
    seen = {np.asarray(e["commands"]).tobytes() for e in cap["evaluated_candidates"]}
    seen.update(np.asarray(c["commands"]).tobytes() for c in registration["cases"])
    gates = preflight(
        command,
        x,
        previous,
        np.asarray(runtime.control["input_min"]),
        np.asarray(runtime.control["input_max"]),
        lower,
        upper,
        spec["radius"],
        seen,
    )
    require(all(gates.values()), "Original command preflight; no clipping or resampling")
    cases = [
        dict(role="anchor", commands=x),
        dict(role="corrected_rotation", commands=command, admission=gates),
        dict(role="anchor_repeat", commands=x.copy()),
    ]
    result = runtime.run(
        cases,
        anchor,
        CONFIG,
        PLAN,
        [__file__, CONFIG, PLAN],
        dict(
            experiment="corrected_rotation_probe",
            response_count_per_side=3,
            total_query_budget=1874,
            total_inner_step_budget=1800,
            qp_result_ref=receipt,
            external_qp_solves=1,
            external_nnls_fit_attempts=qp["nnls_fit_attempts"],
            new_causal_secant_updates=1,
            predicted_trial_safety=np.asarray(anchor["safety_slack"]) + models["Js"] @ (command - x).ravel(),
        ),
    )
    require(result["passed"], "Full independent response protocol")


if __name__ == "__main__":
    main()
