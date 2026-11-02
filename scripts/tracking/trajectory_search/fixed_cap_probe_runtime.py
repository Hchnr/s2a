"""Shared finite full-objective QP and response orchestration."""

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.research.trajectory_search.cap_distance_objective import CapDistanceObjective
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import reference, require
from s2a.research.trajectory_search.state_vector_dual_recovery import objective_divisor, solve_recovered_epigraph_qp
from scripts.tracking.trajectory_search.cap_distance_audit_policy import CapDistanceAuditPolicy
from scripts.tracking.trajectory_search.probe_corrected_cap_distance import fixed_trials
from scripts.tracking.trajectory_search.state_vector_dual_recovery_audit import _recovery_record, _scale_record
from scripts.tracking.trajectory_search.state_vector_gn_search_audit import _error, _pose
from scripts.verification.trajectory_search.accepted_point_sqp_math_audit import audit_qp_certificate
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare


def run_fixed_cap_probe(runtime, spec, config, plan, anchor, models, cap, model_ref, entry_path, experiment):
    """Audit one QP, freeze five scales, then independently replay every response."""
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
    prereg = publish_result(
        runtime.output / "qp_registration.json",
        dict(
            spec=spec,
            config_ref=reference(config),
            plan_ref=reference(plan),
            qa_ref=runtime.qa_ref,
            input_refs=list(runtime.refs.values()),
            B=B,
            g=g,
            A=A,
            b=b,
            initial_t=problem["initial_t"],
            new_model_fits=0,
            model_ref=model_ref,
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
        runtime.output / "qp_result.json",
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
        config,
        plan,
        [entry_path, __file__, config, plan],
        dict(
            experiment=experiment,
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

    return r
