"""Independent first061 candidate request math/pairing; no task qualification."""

import os
from pathlib import Path

from verify_wam_clean_training_capture import Evidence, paired_initial, require
from verify_wam_terminal_development_points import audit_points, head_parameters

from s2a.clean_training_runtime import publish, reference


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    root = Path("outputs/terminal_development_20261025").resolve()
    require(not (root / "first_candidate_request_witness.json").exists(), "Immutable observation required")
    e = Evidence()
    pref = reference(root / "prepared.json")
    p = e.json(pref)
    heads = tuple(head_parameters(e.json(ref)) for ref in (p["baseline_params_ref"], p["candidate_params_ref"]))
    directory = root / "development/service/requests/075_task0_init0_s42_terminal058_q00"
    record = e.json(reference(directory / "completed.json"))
    require(
        record["complete"] is True and record["global_request_index"] == 75 and record["variant"] == "terminal058",
        "Complete fixed first candidate request required",
    )
    ledger = e.json(record["compact_ledger_ref"])
    header = e.json(ledger["request_ref"])
    trace = e.json(record["trace_ref"])
    _, count, changed = audit_points(e, ledger, header, directory, trace["steps"], head=heads, purpose="evaluation")
    raw_directory = root / "development/service/requests/000_task0_init0_s42_raw_q00"
    raw = e.json(reference(raw_directory / "completed.json"))
    old_initial, new_initial = (e.json(r["initial_ref"]) for r in (raw, record))
    old_arrays = e.arrays(old_initial["arrays"], raw_directory / "initial.npz")
    new_arrays = e.arrays(new_initial["arrays"], directory / "initial.npz")
    paired_initial(new_initial, new_arrays, old_initial, old_arrays)
    require(raw["image_ref"]["sha256"] == record["image_ref"]["sha256"], "First physical current PNG differs")
    e.finish()
    publish(
        root / "first_candidate_request_witness.json",
        dict(
            passed=True,
            complete=True,
            partial=True,
            prepared_ref=pref,
            nodes_verified=count,
            changed_active_values=changed,
            first_current_pair_exact=True,
            input_sha256=e.inputs,
            source_refs=[
                reference(Path(__file__)),
                reference(Path("scripts/verify_wam_terminal_development_points.py")),
            ],
            new_model_calls=0,
            new_physics_steps=0,
            fresh_cases=0,
            candidate_stability_qualified=False,
            boundary="One sealed current request only; full12/900,31500 nodes,7200 endpoints and all lifecycle proofs remain required.",
        ),
    )


if __name__ == "__main__":
    main()
