"""One full cap-distance QP with frozen safety correction and five trial scales."""

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.research.trajectory_search.cap_distance_objective import CapDistanceObjective
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, read, reference, require
from s2a.research.trajectory_search.state_vector_dual_recovery import objective_divisor, solve_recovered_epigraph_qp
from scripts.tracking.trajectory_search.cap_distance_audit_policy import CapDistanceAuditPolicy
from scripts.tracking.trajectory_search.probe_rotation_constraints import preflight, reconstruct_models
from scripts.tracking.trajectory_search.response_probe_runtime import ResponseProbeRuntime
from scripts.tracking.trajectory_search.state_vector_dual_recovery_audit import _recovery_record, _scale_record
from scripts.tracking.trajectory_search.state_vector_gn_search_audit import _error, _pose
from scripts.verification.trajectory_search.accepted_point_sqp_math_audit import audit_qp_certificate
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare

OUTPUT = REPO / "outputs/corrected_cap_distance_probe_20261102"
CONFIG = REPO / "configs/tracking/trajectory_search/corrected_cap_distance_probe_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/corrected_cap_distance_probe_20261102.md"
QA = REPO / "outputs/qa/corrected_cap_distance_probe_20261102/result.json"
ALPHAS = [1.0, 0.5, 0.25, 0.125, 0.0625]


def fixed_trials(x, direction, previous, lo, hi, lower, upper, seen):
    x, direction = np.asarray(x), np.asarray(direction)
    require(x.shape == direction.shape == (12, 6), "Complete H12 direction")
    require(np.isfinite(x).all() and np.isfinite(direction).all(), "Finite commands")
    seen = set(seen)
    seen.add(x.tobytes())
    cases = [dict(role="anchor", commands=x.copy(), alpha=0.0)]
    for alpha in ALPHAS:
        command = x + alpha * direction
        gates = preflight(command, x, previous, lo, hi, lower, upper, 0.005, seen)
        require(all(gates.values()), "Every frozen scale passes original command gates without clipping")
        cases.append(dict(role="full_cap_trial", commands=command, alpha=alpha, admission=gates))
        seen.add(command.tobytes())
    cases.append(dict(role="anchor_repeat", commands=x.copy(), alpha=0.0))
    return cases


def main():
    spec = read(reference(CONFIG))
    require(
        spec
        == dict(
            seed=42,
            alphas=ALPHAS,
            radius=0.005,
            qp_budget=1,
            qp_iteration_budget=100,
            nnls_fit_budget=1,
            response_count_per_side=7,
            total_query_budget=4370,
            total_inner_step_budget=4200,
            model_budget=2,
            timeout_seconds=600,
            automatic_retry=False,
        ),
        "Fixed full-objective probe",
    )
    runtime = ResponseProbeRuntime(OUTPUT, QA)
    prior = runtime.load(
        reference(REPO / "outputs/corrected_rotation_probe_20261102/result.json"),
        "177b188ec11af38829466466db9ef768263c6d40ef7c32a0ecc7f9f9bc2351f6",
    )
    require(prior["passed"] and prior["complete"] and prior["sources_unchanged"], "Closed corrected model")
    closed = runtime.load(prior["registration_ref"])
    old_qp = runtime.load(closed["detail"]["qp_result_ref"])
    old_matrix = runtime.load(old_qp["registration_ref"])
    cap_result = runtime.load(
        reference(REPO / "outputs/cap_distance_gn_20261102/result.json"),
        "80f58cde84778d1e3b2f7ff3ca0b85c5414419da976a057710421d23f03012e9",
    )
    cap = runtime.load(cap_result["producer_final_ref"])["answer"]
    stencil = runtime.load(
        reference(REPO / "outputs/replay_order_stencil_20261102/stencil.json"),
        "0d447082a50a2a8425c438a0a7cfe8a4a9742cb53bba445e015e49db6f6ecc3c",
    )
    anchor = runtime.load(reference(REPO / "outputs/corrected_rotation_probe_20261102/production/response_00.json"))
    models = reconstruct_models(cap, stencil, 58)
    models["Js"] = np.asarray(old_matrix["A"])[:4369, :72].copy()
    previous = np.asarray(runtime.source["previous_command"])[:6]
    lower, upper = (np.asarray(cap[k]) for k in ("global_lower", "global_upper"))
    ceiling = np.asarray(cap["pose_ceiling"])
    cfg = cap["config"]
    problem = CapDistanceObjective.problem(
        anchor, models["Js"], models["Jq"], models["Je"], previous, lower, upper, ceiling, spec["radius"], "pose", cfg
    )
    B, g, A, b = CapDistanceAuditPolicy.problem(
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
        cfg,
    )
    for key, value in dict(B=B, g=g, A=A, b=b).items():
        np.testing.assert_array_equal(problem[key], value)
    np.testing.assert_array_equal(A, old_matrix["A"])
    np.testing.assert_array_equal(b, old_matrix["b"])
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
            new_model_fits=0,
            safety_model_ref=old_qp["registration_ref"],
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

    receipt = publish_result(
        OUTPUT / "qp_result.json",
        dict(
            registration_ref=prereg,
            qp=qp,
            certificate=audit_qp_certificate(qp, B, g, A, b),
            scale=_scale_record(qp, B, g, A, b, divisor, "pose", compare),
            recovery=_recovery_record(qp, B, g, A, b, "pose", compare),
            qp_solves=1,
            nnls_fit_attempts=qp["nnls_fit_attempts"],
            control_qualified=False,
        ),
    )
    require(qp["qualified"] and qp["nnls_fit_attempts"] <= 1, "Qualified single QP within budget")
    x = np.asarray(anchor["commands"])
    direction = np.asarray(qp["point"])[:72].reshape(12, 6)
    cases = fixed_trials(
        x,
        direction,
        previous,
        np.asarray(runtime.control["input_min"]),
        np.asarray(runtime.control["input_max"]),
        lower,
        upper,
        {np.asarray(e["commands"]).tobytes() for e in cap["evaluated_candidates"]},
    )
    r = runtime.run(
        cases,
        anchor,
        CONFIG,
        PLAN,
        [__file__, CONFIG, PLAN],
        dict(
            experiment="corrected_cap_distance_probe",
            response_count_per_side=7,
            total_query_budget=4370,
            total_inner_step_budget=4200,
            qp_result_ref=receipt,
            external_qp_solves=1,
            external_nnls_fit_attempts=qp["nnls_fit_attempts"],
            new_model_fits=0,
            linear_safety_predictions=[
                np.asarray(anchor["safety_slack"]) + models["Js"] @ (np.asarray(c["commands"]) - x).ravel()
                for c in cases
            ],
            predicted_cap_reductions=[
                float(
                    -(g @ ((np.asarray(c["commands"]) - x).ravel()))
                    - 0.5 * np.sum((problem["cap_model"]["jacobian"] @ (np.asarray(c["commands"]) - x).ravel()) ** 2)
                )
                for c in cases
            ],
        ),
    )
    require(r["passed"], "Full independent response audit")


if __name__ == "__main__":
    main()
