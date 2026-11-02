"""Fixed shortening/small-step cases on the closed rotation-probe direction."""

from pathlib import Path

import numpy as np

from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO, read, reference, require
from scripts.tracking.trajectory_search.probe_rotation_constraints import preflight
from scripts.tracking.trajectory_search.response_probe_runtime import ResponseProbeRuntime

OUTPUT = REPO / "outputs/rotation_direction_locality_20261102"
CONFIG = REPO / "configs/tracking/trajectory_search/rotation_direction_locality_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/rotation_direction_locality_20261102.md"
QA = REPO / "outputs/qa/rotation_direction_locality_final_20261102/result.json"


def validate_spec(spec):
    expected = dict(
        seed=42,
        backtracking_alphas=[0.5, 0.25, 0.125, 0.0625],
        small_step_inf=[1e-4, 5e-5],
        radius=0.005,
        response_count_per_side=8,
        total_response_budget=16,
        total_query_budget=4994,
        total_inner_step_budget=4800,
        model_budget=2,
        qp_budget=0,
        nnls_fit_budget=0,
        timeout_seconds=600,
        automatic_retry=False,
        tool_policy="native_solve",
    )
    require(spec == expected, "Fixed locality sample set, scales and budget")


def build_cases(anchor, direction, spec, previous, input_min, input_max, lower, upper):
    validate_spec(spec)
    x, d = np.asarray(anchor, dtype=float), np.asarray(direction, dtype=float)
    require(
        x.shape == d.shape == (12, 6) and np.isfinite(x).all() and np.isfinite(d).all(),
        "Finite complete trajectory/direction",
    )
    norm = float(np.max(abs(d)))
    require(norm > max(spec["small_step_inf"]), "Original direction exceeds both fixed local scales")
    alphas = spec["backtracking_alphas"] + [h / norm for h in spec["small_step_inf"]]
    cases = [dict(role="anchor", alpha=0.0, commands=x.copy(), actual_step_inf=0.0)]
    seen = {x.tobytes()}
    for index, alpha in enumerate(alphas):
        command = x + alpha * d
        gates = preflight(command, x, previous, input_min, input_max, lower, upper, spec["radius"], seen)
        require(all(gates.values()), "Every fixed case must be admitted without clipping or resampling")
        cases.append(
            dict(
                role="backtracking" if index < 4 else "small_step",
                alpha=float(alpha),
                commands=command,
                requested_step_inf=None if index < 4 else spec["small_step_inf"][index - 4],
                actual_step_inf=float(np.max(abs(command - x))),
                admission=gates,
            )
        )
        seen.add(command.tobytes())
    cases.append(dict(role="anchor_repeat", alpha=0.0, commands=x.copy(), actual_step_inf=0.0))
    require(len(cases) == 8, "Registered eight cases including initial repeat")
    return cases


def main():
    spec = read(reference(CONFIG))
    validate_spec(spec)
    runtime = ResponseProbeRuntime(OUTPUT, QA)
    prior_ref = reference(REPO / "outputs/rotation_constraint_probe_20261102/result.json")
    prior = runtime.load(prior_ref, "7d5bd9502573cf122208a9c146c3baac8aaa62991f03688716fd58841238bdc8")
    require(
        prior["complete"] and prior["passed"] and prior["sources_unchanged"] and prior["trial_evaluated"],
        "Closed unique direction with exact independent response",
    )
    entries = runtime.load(prior["entries_ref"])
    registered = runtime.load(prior["registration_ref"])
    cap_result = runtime.load(
        registered["cap_result_ref"], "80f58cde84778d1e3b2f7ff3ca0b85c5414419da976a057710421d23f03012e9"
    )
    cap = runtime.load(cap_result["producer_final_ref"])["answer"]
    x = np.asarray(entries[0]["commands"])
    d = np.asarray(entries[1]["commands"]) - x
    c = runtime.control
    s = runtime.source
    cases = build_cases(
        x,
        d,
        spec,
        np.asarray(s["previous_command"])[:6],
        np.asarray(c["input_min"]),
        np.asarray(c["input_max"]),
        np.asarray(cap["global_lower"]),
        np.asarray(cap["global_upper"]),
    )
    A = np.asarray(registered["A"])
    base = np.asarray(entries[0]["safety_slack"])
    detail = dict(
        experiment="rotation_direction_locality",
        response_count_per_side=8,
        total_query_budget=4994,
        total_inner_step_budget=4800,
        prior_probe_ref=prior_ref,
        historical_alpha1_ref=prior["entries_ref"],
        original_direction=d,
        original_direction_inf=float(np.max(abs(d))),
        linear_safety_predictions=[base + A[:4369, :72] @ (np.asarray(case["commands"]) - x).ravel() for case in cases],
    )
    result = runtime.run(
        cases,
        entries[0],
        CONFIG,
        PLAN,
        [
            Path(__file__),
            REPO / "scripts/tracking/trajectory_search/response_probe_runtime.py",
            REPO / "scripts/tracking/trajectory_search/probe_rotation_constraints.py",
        ],
        detail,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
