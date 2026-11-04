"""Fixed responses for three qualified directions skipped by the progress gate."""

import numpy as np

from s2a.research.trajectory_search.reserve_aligned_cap_objective import ReserveAlignedCapObjective
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, read, reference, require
from s2a.research.trajectory_search.state_vector_dual_recovery import objective_divisor
from scripts.tracking.trajectory_search.probe_rotation_constraints import preflight, reconstruct_models
from scripts.tracking.trajectory_search.reserve_aligned_cap_audit import ReserveAlignedCapAuditPolicy
from scripts.tracking.trajectory_search.response_probe_runtime import ResponseProbeRuntime
from scripts.tracking.trajectory_search.state_vector_dual_recovery_audit import _recovery_record, _scale_record
from scripts.tracking.trajectory_search.state_vector_gn_search_audit import _error, _pose
from scripts.verification.trajectory_search.accepted_point_sqp_math_audit import audit_qp_certificate
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare

OUTPUT = REPO / "outputs/deferred_cap_directions_20261102"
CONFIG = REPO / "configs/tracking/trajectory_search/deferred_cap_directions_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/deferred_cap_directions_20261102.md"
QA = REPO / "outputs/qa/deferred_cap_directions_20261102/result.json"


def select_cases(answer, spec, previous, lo, hi):
    require(
        spec
        == dict(
            seed=42,
            stage_indices=[9, 10, 11],
            anchor_index=18,
            response_count_per_side=5,
            total_response_budget=10,
            total_inner_step_budget=3000,
            total_query_budget=3122,
            model_budget=2,
            qp_budget=0,
            nnls_fit_budget=0,
            new_model_fits=0,
            timeout_seconds=600,
            automatic_retry=False,
        ),
        "Exact fixed directions and budgets",
    )
    anchor = answer["evaluated_candidates"][18]
    x = np.asarray(anchor["commands"])
    require(x.shape == (12, 6) and np.isfinite(x).all(), "Complete finite anchor")
    seen = {np.asarray(e["commands"]).tobytes() for e in answer["evaluated_candidates"]}
    cases = [dict(role="anchor", commands=x.copy())]
    for index in spec["stage_indices"]:
        stage = answer["stages"][index]
        require(
            stage["mode"] == "pose" and stage["qp"]["qualified"] and stage["anchor_before_index"] == 18,
            "Same fixed anchor and qualified pose QP",
        )
        t = stage["attempts"][0]
        require(
            t["alpha"] == 1.0
            and t["response_index"] is None
            and t["acceptance_reason"] == "predicted_reduction_insufficient"
            and 0 < t["predicted_reduction"] < 1e-4
            and all(t["admission"].values()),
            "Positive deferred first direction only",
        )
        command = np.asarray(t["commands"])
        require(command.shape == (12, 6) and np.isfinite(command).all(), "Complete finite deferred command")
        np.testing.assert_array_equal(command, x + np.asarray(stage["qp"]["point"])[:72].reshape(12, 6))
        gates = preflight(
            command,
            x,
            previous,
            lo,
            hi,
            np.asarray(answer["global_lower"]),
            np.asarray(answer["global_upper"]),
            stage["radius_before"],
            seen,
        )
        require(all(gates.values()), "Original command gates without clipping or resampling")
        cases.append(
            dict(
                role="deferred_trial",
                stage_index=index,
                commands=command,
                admission=gates,
                historical_predicted_reduction=t["predicted_reduction"],
            )
        )
        seen.add(command.tobytes())
    cases.append(dict(role="anchor_repeat", commands=x.copy()))
    return anchor, cases


def main():
    spec = read(reference(CONFIG))
    runtime = ResponseProbeRuntime(OUTPUT, QA)
    prior_ref = reference(REPO / "outputs/reserve_safety_search_20261102/safety_feedback/result.json")
    prior = runtime.load(prior_ref, "93deb627f6bd51f5830255a05aaab58d835c58492cb9f10a84af35a62490b9ef")
    require(
        prior["passed"] and prior["complete"] and prior["sources_unchanged"] and prior["exact_native_responses"],
        "Closed exact native source",
    )
    answer = runtime.load(prior["producer_final_ref"])["answer"]
    stencil = runtime.load(
        reference(REPO / "outputs/anchor85_coupled_stencil_20261102/stencil.json"),
        "f4bc18043a6c1de5ae40e3c266eb7b5cb4cbe1ea9ac9af137ecf6e2058cac417",
    )
    previous = np.asarray(runtime.source["previous_command"])[:6]
    anchor, cases = select_cases(
        answer, spec, previous, np.asarray(runtime.control["input_min"]), np.asarray(runtime.control["input_max"])
    )
    models = reconstruct_models(answer, stencil, 9)
    cfg = answer["config"]
    checks = []
    for index in spec["stage_indices"]:
        current = reconstruct_models(answer, stencil, index)
        for key in models:
            np.testing.assert_array_equal(models[key], current[key])
        stage = answer["stages"][index]
        args = (
            anchor,
            models["Js"],
            models["Jq"],
            models["Je"],
            previous,
            np.asarray(answer["global_lower"]),
            np.asarray(answer["global_upper"]),
            np.asarray(answer["pose_ceiling"]),
            stage["radius_before"],
            "pose",
            cfg,
        )
        p = ReserveAlignedCapObjective.problem(*args)
        B, g, A, b = ReserveAlignedCapAuditPolicy.problem(
            np.asarray(anchor["commands"]), np.asarray(anchor["safety_slack"]), _pose(anchor), _error(anchor), *args[1:]
        )
        for key, value in dict(B=B, g=g, A=A, b=b).items():
            np.testing.assert_array_equal(p[key], value)

        def compare(n, a, b):
            return semantic_compare(n, a, b, 1e-10)

        checks.append(
            dict(
                stage=index,
                certificate=audit_qp_certificate(stage["qp"], B, g, A, b),
                scaling=_scale_record(
                    stage["qp"], B, g, A, b, objective_divisor(p, stage["radius_before"], "pose"), "pose", compare
                ),
                recovery=_recovery_record(stage["qp"], B, g, A, b, "pose", compare),
            )
        )
    result = runtime.run(
        cases,
        anchor,
        CONFIG,
        PLAN,
        [__file__, CONFIG, PLAN],
        dict(
            experiment="deferred_cap_directions",
            response_count_per_side=5,
            total_query_budget=3122,
            total_inner_step_budget=3000,
            prior_search_ref=prior_ref,
            closed_qp_reconstructions=checks,
            new_qp_solves=0,
            new_nnls_fits=0,
            new_model_fits=0,
        ),
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
