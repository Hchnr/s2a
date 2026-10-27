"""First sealed060 candidate versus qualified052 history, no future states."""

import os
from pathlib import Path

import numpy as np
from verify_wam_clean_training_capture import Evidence, require

from s2a.clean_training_runtime import publish, reference
from s2a.terminal_history_comparison import INPUTS, OUTPUTS, compare_point


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    root = Path("outputs/terminal_identity_state_validation_20261025").resolve()
    require(not (root / "first_candidate_history_witness.json").exists(), "Immutable output required")
    e = Evidence()
    witness_ref = reference(root / "first_candidate_numerical_witness.json")
    witness = e.json(witness_ref)
    require(
        witness["passed"] is witness["complete"] is witness["partial"] is True and witness["nodes_verified"] == 35,
        "Sealed first candidate witness required",
    )
    p = e.json(witness["prepared_ref"])
    row = p["sampling_entries"][4]
    oldroot = Path("outputs/synchronous_state_probe052_continuation_20261024")
    oldnum = e.json(reference(oldroot / "numerical_verified.json"))
    require(
        all(oldnum[k] is True for k in ("passed", "complete", "numerical_contract_passed", "lifecycle_passed")),
        "Qualified052 history required",
    )
    previous = e.json(reference(oldroot / "state_sampling/result.json"))
    matches = [
        g
        for g in previous["generations"]
        if g["variant"] == "probe052"
        and g["anchor_name"] == row["anchor_name"]
        and g["sampling_seed"] == row["sampling_seed"]
    ]
    require(len(matches) == 1, "Exactly one original052 case required")
    old = matches[0]
    require(
        any(g["response_ref"] == old["response_ref"] for g in oldnum["generated_refs"]),
        "Original052 case not numerically qualified",
    )
    old_request = e.json(old["request_ref"])
    new_request = dict(row["request"])
    old_request.pop("name")
    new_request.pop("name")
    require(old_request == new_request, "Current request/seed differs")
    a, b = e.json(witness["ledger_ref"]), e.json(old["compact_ledger_ref"])
    require(a["complete"] and b["complete"] and len(a["points"]) == len(b["points"]) == 35, "Full35 captures required")
    results = []
    for step, (ar, br) in enumerate(zip(a["points"], b["points"], strict=True)):
        am, bm = e.json(ar), e.json(br)
        require(
            am["metadata"]["step_index"] == bm["metadata"]["step_index"] == step
            and am["metadata"]["sigma_actual"] == bm["metadata"]["sigma_actual"],
            "Actual noise node differs",
        )
        arrays = []
        for point in (am, bm):
            with np.load(e.path(point["arrays_ref"]), allow_pickle=False) as z:
                arrays.append({k: z[k].copy() for k in INPUTS + OUTPUTS})
        results.append(compare_point(*arrays, step))
    e.finish()
    publish(
        root / "first_candidate_history_witness.json",
        dict(
            passed=all(r["passed"] for r in results),
            complete=True,
            partial=True,
            numerical_witness_ref=witness_ref,
            requests=1,
            nodes=35,
            steps=results,
            input_sha256=e.inputs,
            source_refs=[reference(Path(__file__)), reference(Path("src/s2a/terminal_history_comparison.py"))],
            boundary="One sealed request only; no full059/060 lifecycle or task-performance qualification.",
            new_model_calls=0,
            new_fits=0,
            physics_steps=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
