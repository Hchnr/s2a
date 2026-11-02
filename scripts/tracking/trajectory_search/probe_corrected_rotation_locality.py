"""Fixed locality responses on the unique closed corrected-rotation direction."""

import numpy as np

from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, read, reference, require
from scripts.tracking.trajectory_search.probe_rotation_direction_locality import build_cases, validate_spec
from scripts.tracking.trajectory_search.response_probe_runtime import ResponseProbeRuntime

OUTPUT = REPO / "outputs/corrected_rotation_locality_20261102"
CONFIG = REPO / "configs/tracking/trajectory_search/corrected_rotation_locality_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/corrected_rotation_locality_20261102.md"
QA = REPO / "outputs/qa/corrected_rotation_locality_20261102/result.json"


def main():
    spec = read(reference(CONFIG))
    validate_spec(spec)
    runtime = ResponseProbeRuntime(OUTPUT, QA)
    prior_ref = reference(REPO / "outputs/corrected_rotation_probe_20261102/result.json")
    prior = runtime.load(prior_ref, "177b188ec11af38829466466db9ef768263c6d40ef7c32a0ecc7f9f9bc2351f6")
    require(prior["passed"] and prior["complete"] and prior["sources_unchanged"], "Closed corrected probe")
    reg = runtime.load(prior["registration_ref"])
    qp_result = runtime.load(reg["detail"]["qp_result_ref"])
    qp_reg = runtime.load(qp_result["registration_ref"])
    anchor = runtime.load(reference(REPO / "outputs/corrected_rotation_probe_20261102/production/response_00.json"))
    trial = runtime.load(reference(REPO / "outputs/corrected_rotation_probe_20261102/production/response_01.json"))
    # The original unchanged cap search still defines every global command bound.
    old_ref = reference(REPO / "outputs/cap_distance_gn_20261102/result.json")
    old = runtime.load(old_ref, "80f58cde84778d1e3b2f7ff3ca0b85c5414419da976a057710421d23f03012e9")
    cap = runtime.load(old["producer_final_ref"])["answer"]
    x = np.asarray(anchor["commands"])
    d = np.asarray(trial["commands"]) - x
    np.testing.assert_array_equal(x, reg["cases"][0]["commands"])
    np.testing.assert_array_equal(x + d, reg["cases"][1]["commands"])
    cases = build_cases(
        x,
        d,
        spec,
        np.asarray(runtime.source["previous_command"])[:6],
        np.asarray(runtime.control["input_min"]),
        np.asarray(runtime.control["input_max"]),
        np.asarray(cap["global_lower"]),
        np.asarray(cap["global_upper"]),
    )
    Js = np.asarray(qp_reg["A"])[:4369, :72]
    base = np.asarray(anchor["safety_slack"])
    result = runtime.run(
        cases,
        anchor,
        CONFIG,
        PLAN,
        [__file__, CONFIG, PLAN],
        dict(
            experiment="corrected_rotation_locality",
            response_count_per_side=8,
            total_query_budget=4994,
            total_inner_step_budget=4800,
            prior_probe_ref=prior_ref,
            original_direction=d,
            original_direction_inf=float(np.max(abs(d))),
            safety_model_ref=qp_result["registration_ref"],
            historical_alpha1_ref=reference(
                REPO / "outputs/corrected_rotation_probe_20261102/production/response_01.json"
            ),
            linear_safety_predictions=[base + Js @ (np.asarray(c["commands"]) - x).ravel() for c in cases],
        ),
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
