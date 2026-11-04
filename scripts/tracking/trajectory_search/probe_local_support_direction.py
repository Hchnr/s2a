"""Frozen terminal-support direction: numerical nonlinear observations, never dispatch."""

import numpy as np

from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, read, reference, require
from scripts.tracking.trajectory_search.analyze_local_tracking_support import (
    audit_support,
    box_corrected_bound,
    support_rows,
)
from scripts.tracking.trajectory_search.probe_rotation_constraints import reconstruct_models
from scripts.tracking.trajectory_search.response_probe_runtime import ResponseProbeRuntime

OUTPUT = REPO / "outputs/local_support_direction_probe_20261102"
CONFIG = REPO / "configs/tracking/trajectory_search/local_support_direction_probe_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/local_support_direction_probe_20261102.md"
QA = REPO / "outputs/qa/local_support_direction_probe_20261102/result.json"


def select_cases(answer, lp, spec, previous, lo, hi):
    require(
        spec
        == dict(
            seed=42,
            source_lp_sha256="6ff37e4cf90c1763cdffe0c961c6078c8838371351909f9b38b1470a2630b09b",
            anchor_index=23,
            group=23,
            reserve_fraction=0.9,
            alphas=[1.0, 0.5, 0.25],
            response_count_per_side=5,
            total_response_budget=10,
            total_inner_step_budget=3000,
            total_query_budget=3122,
            model_budget=2,
            new_lp_budget=0,
            new_qp_budget=0,
            new_nnls_budget=0,
            timeout_seconds=600,
            automatic_retry=False,
        ),
        "Exact fixed direction and budgets",
    )
    require(
        lp["success"] and lp["anchor_index"] == 23 and lp["group"] == 23 and lp["reserve_fraction"] == 0.9,
        "Only retained final terminal support solution",
    )
    anchor = answer["evaluated_candidates"][23]
    x, d = np.asarray(anchor["commands"]), np.asarray(lp["point"])
    require(
        x.shape == (12, 6) and d.shape == (72,) and np.isfinite(x).all() and np.isfinite(d).all(),
        "Full finite trajectory and LP direction",
    )
    require(np.any(d != 0), "Nonzero saved direction")
    cases = [dict(role="anchor", commands=x.copy(), alpha=0.0)]
    seen = {x.tobytes()}
    for alpha in spec["alphas"]:
        command = x + alpha * d.reshape(12, 6)
        require(
            np.all(command >= np.asarray(lo) - 1e-12) and np.all(command <= np.asarray(hi) + 1e-12),
            "Original input bounds",
        )
        require(
            np.all(command >= np.asarray(answer["global_lower"]) - 1e-12)
            and np.all(command <= np.asarray(answer["global_upper"]) + 1e-12),
            "Original root domain",
        )
        require(np.all(np.abs(np.diff(np.vstack((previous, command)), axis=0)) <= 0.25 + 1e-12), "Original rate bound")
        require(command.tobytes() not in seen, "Distinct fixed diagnostic command")
        cases.append(dict(role="support_direction", commands=command, alpha=alpha))
        seen.add(command.tobytes())
    cases.append(dict(role="anchor_repeat", commands=x.copy(), alpha=0.0))
    return anchor, cases


def main():
    spec = read(reference(CONFIG))
    runtime = ResponseProbeRuntime(OUTPUT, QA)
    support = runtime.load(
        reference(REPO / "outputs/local_tracking_support_20261102/result.json"),
        "f029085ade6a4c846fffb50ce0eba294b0a87cbbf2aedc5cf694192555d90203",
    )
    require(
        support["passed"] and support["sources_unchanged"] and support["inputs_unchanged"], "Closed support analysis"
    )
    lp_ref = reference(REPO / "outputs/local_tracking_support_20261102/lp_48.json")
    lp = runtime.load(lp_ref, spec["source_lp_sha256"])
    old = runtime.load(
        reference(REPO / "outputs/small_progress_search_20261102/small_progress/result.json"),
        "844c4e05a24b41dda33e36aac180bb421afa73e63f88f8eb70b213615f6e4c5a",
    )
    answer = runtime.load(old["producer_final_ref"])["answer"]
    repaired = runtime.load(
        reference(REPO / "outputs/small_progress_native_repair_20261102/result.json"),
        "52a41511b116e958a3bb7ba43adbfaf1485eb351be1e88a5349e4a79eae0ac90",
    )
    require(
        repaired["passed"] and repaired["exact_native_responses"] and repaired["verified_responses"] == 36,
        "Original failed receipt independently completed, never rewritten",
    )
    stencil = runtime.load(
        reference(REPO / "outputs/anchor85_coupled_stencil_20261102/stencil.json"),
        "f4bc18043a6c1de5ae40e3c266eb7b5cb4cbe1ea9ac9af137ecf6e2058cac417",
    )
    previous = np.asarray(runtime.source["previous_command"])[:6]
    anchor, cases = select_cases(answer, lp, spec, previous, runtime.control["input_min"], runtime.control["input_max"])
    models = reconstruct_models(answer, stencil, answer["total_stages"])
    A, b = support_rows(
        anchor, models["Js"], previous, answer["global_lower"], answer["global_upper"], answer["config"], 0.9
    )
    original_rows = runtime.load(lp["constraints_ref"])
    np.testing.assert_array_equal(A, original_rows["A"])
    np.testing.assert_array_equal(b, original_rows["b"])
    E = (np.asarray(anchor["tcp_errors"]) / np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])).ravel()
    normal = E[-3:] / np.linalg.norm(E[-3:])
    c = normal @ models["Je"][-3:]
    np.testing.assert_array_equal(c, lp["c"])
    np.testing.assert_array_equal(normal, lp["normal"])
    certificate = audit_support(A, b, c, lp["point"], lp["dual"])
    x = np.asarray(anchor["commands"])
    correction = box_corrected_bound(
        A,
        b,
        c,
        np.asarray(lp["dual"]),
        (np.asarray(answer["global_lower"]) - x).ravel(),
        (np.asarray(answer["global_upper"]) - x).ravel(),
    )
    require(
        float(np.linalg.norm(E[-3:]) + correction["bound"]) == lp["support_lower_bound"], "Exact retained support bound"
    )
    predictions = []
    for case in cases:
        step = (case["commands"] - x).ravel()
        error = E + models["Je"] @ step
        predictions.append(
            dict(
                alpha=case["alpha"],
                normalized_error=error.reshape(12, 6),
                safety_slack=np.asarray(anchor["safety_slack"]) + models["Js"] @ step,
                terminal_support=float(normal @ error[-3:]),
                terminal_rotation_norm=float(np.linalg.norm(error[-3:])),
            )
        )
    result = runtime.run(
        cases,
        anchor,
        CONFIG,
        PLAN,
        [__file__, CONFIG, PLAN],
        dict(
            experiment="local_support_direction_probe",
            response_count_per_side=5,
            total_query_budget=3122,
            total_inner_step_budget=3000,
            source_lp_ref=lp_ref,
            certificate=certificate,
            corrected_bound=correction,
            normal=normal,
            predictions=predictions,
            new_lp_solves=0,
            new_qp_solves=0,
            new_nnls_fits=0,
            new_model_fits=0,
        ),
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
