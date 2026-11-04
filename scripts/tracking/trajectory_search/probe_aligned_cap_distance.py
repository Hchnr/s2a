"""Full cap-distance direction from freshly collected coupled local models."""

import numpy as np

from s2a.research.trajectory_search.reserve_aligned_cap_objective import POLICY, ReserveAlignedCapObjective
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, read, reference, require
from scripts.tracking.trajectory_search.fixed_cap_probe_runtime import run_fixed_cap_probe
from scripts.tracking.trajectory_search.probe_corrected_cap_distance import ALPHAS
from scripts.tracking.trajectory_search.reserve_aligned_cap_audit import ReserveAlignedCapAuditPolicy
from scripts.tracking.trajectory_search.response_probe_runtime import ResponseProbeRuntime

OUTPUT = REPO / "outputs/aligned_cap_distance_probe_20261102"
CONFIG = REPO / "configs/tracking/trajectory_search/aligned_cap_distance_probe_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/aligned_cap_distance_probe_20261102.md"
QA = REPO / "outputs/qa/aligned_cap_distance_probe_20261102/result.json"


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
            problem_policy=POLICY,
        ),
        "Fixed refreshed-model probe",
    )
    runtime = ResponseProbeRuntime(OUTPUT, QA)
    qualified = runtime.load(
        reference(REPO / "outputs/anchor85_coupled_stencil_20261102/result.json"),
        "bee4d9399fbbda691218c8f4ec9f9739f0026d072c1d9c51f3ab097bee7c79ce",
    )
    require(
        qualified["passed"]
        and qualified["complete"]
        and qualified["sources_unchanged"]
        and qualified["exact_native_responses"],
        "Complete refreshed-model qualification",
    )
    model_ref = qualified["stencil_ref"]
    model = runtime.load(model_ref, "f4bc18043a6c1de5ae40e3c266eb7b5cb4cbe1ea9ac9af137ecf6e2058cac417")
    old = runtime.load(
        reference(REPO / "outputs/cap_distance_gn_20261102/result.json"),
        "80f58cde84778d1e3b2f7ff3ca0b85c5414419da976a057710421d23f03012e9",
    )
    cap = runtime.load(old["producer_final_ref"])["answer"]
    anchor = model["base"]
    np.testing.assert_array_equal(anchor["commands"], cap["evaluated_candidates"][85]["commands"])
    models = {k: np.asarray(model["matrices"][k]) for k in ("Js", "Jq", "Je")}
    run_fixed_cap_probe(
        runtime,
        spec,
        CONFIG,
        PLAN,
        anchor,
        models,
        cap,
        model_ref,
        __file__,
        "aligned_cap_distance_probe",
        objective=ReserveAlignedCapObjective(),
        objective_audit=ReserveAlignedCapAuditPolicy(),
    )


if __name__ == "__main__":
    main()
