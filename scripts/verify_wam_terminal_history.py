"""Compare all058 neural histories to historical052, only after full060 audit.

No future states, fitting or model calls. A mismatch is preserved as evidence;
this diagnostic cannot relax the original independent numerical qualification.
"""

import os
from pathlib import Path

import numpy as np
from verify_wam_clean_training_capture import Evidence, require

from s2a.clean_training_runtime import publish, reference
from s2a.terminal_history_comparison import INPUTS, OUTPUTS, compare_point


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    root = Path("outputs/terminal_identity_state_validation_20261025")
    require(not (root / "historical052_comparison.json").exists(), "Immutable output required")
    e = Evidence()
    numerical_ref = reference(root / "numerical_verified.json")
    numerical = e.json(numerical_ref)
    oldroot = Path("outputs/synchronous_state_probe052_continuation_20261024")
    oldnum = e.json(reference(oldroot / "numerical_verified.json"))
    for doc in (numerical, oldnum):
        require(
            all(doc[k] is True for k in ("passed", "complete", "numerical_contract_passed", "lifecycle_passed")),
            "Complete original numerical/lifecycle gate required",
        )
    current = e.json(numerical["sampling_ref"])
    previous = e.json(reference(oldroot / "state_sampling/result.json"))
    current_rows = {
        (g["anchor_name"], g["sampling_seed"]): g for g in current["generations"] if g["variant"] == "terminal058"
    }
    previous_rows = {
        (g["anchor_name"], g["sampling_seed"]): g for g in previous["generations"] if g["variant"] == "probe052"
    }
    require(
        set(current_rows) == set(previous_rows) and len(current_rows) == 48,
        "All48 identical current/seed pairs required",
    )
    for rows, doc in ((current_rows, numerical), (previous_rows, oldnum)):
        admitted = {g["response_ref"]["sha256"] for g in doc["generated_refs"]}
        require(
            all(g["response_ref"]["sha256"] in admitted for g in rows.values()),
            "Unqualified historical/current generation",
        )
    results = []
    for key, row in current_rows.items():
        old = previous_rows[key]
        request, old_request = e.json(row["request_ref"]), e.json(old["request_ref"])
        request.pop("name")
        old_request.pop("name")
        require(request == old_request, "Historical current inputs/request differ")
        a, b = e.json(row["compact_ledger_ref"]), e.json(old["compact_ledger_ref"])
        require(
            a["complete"] and b["complete"] and len(a["points"]) == len(b["points"]) == 35,
            "Full paired captures required",
        )
        steps = []
        for step, (ar, br) in enumerate(zip(a["points"], b["points"], strict=True)):
            am, bm = e.json(ar), e.json(br)
            require(
                am["metadata"]["step_index"] == bm["metadata"]["step_index"] == step
                and am["metadata"]["sigma_actual"] == bm["metadata"]["sigma_actual"],
                "Historical timestep differs",
            )
            arrays = []
            for point in (am, bm):
                with np.load(e.path(point["arrays_ref"]), allow_pickle=False) as z:
                    arrays.append({k: z[k].copy() for k in INPUTS + OUTPUTS})
            steps.append(compare_point(*arrays, step))
        results.append(dict(anchor=key[0], seed=key[1], passed=all(s["passed"] for s in steps), steps=steps))
    e.finish()
    publish(
        root / "historical052_comparison.json",
        dict(
            complete=True,
            passed=all(r["passed"] for r in results),
            numerical_ref=numerical_ref,
            requests=48,
            nodes=1680,
            comparisons=results,
            input_sha256=e.inputs,
            source_refs=[reference(Path(__file__)), reference(Path("src/s2a/terminal_history_comparison.py"))],
            new_model_calls=0,
            new_fits=0,
            physics_steps=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
