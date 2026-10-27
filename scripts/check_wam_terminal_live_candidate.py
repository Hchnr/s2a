"""Partial numerical witness for the first sealed060 candidate; no future labels."""

import os
from pathlib import Path

from verify_wam_clean_training_capture import Evidence, require
from verify_wam_terminal_points import compact_points

from s2a.clean_training_runtime import publish, reference
from s2a.terminal_candidate import candidate_head


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    root = Path("outputs/terminal_identity_state_validation_20261025").resolve()
    require(not (root / "first_candidate_numerical_witness.json").exists(), "Immutable output required")
    e = Evidence()
    prepared_ref = reference(root / "prepared.json")
    p = e.json(prepared_ref)
    row = p["sampling_entries"][4]
    require(row["index"] == 4 and row["variant"] == "terminal058", "Fixed first candidate required")
    policy, params_ref = candidate_head(p["candidate_proof_ref"], read_json=e.json)
    directory = root / "state_sampling" / f"004_{row['name']}"
    ledger_ref = reference(directory / "compact/ledger.json")
    ledger = e.json(ledger_ref)
    require(ledger["complete"] is True and ledger["nodes"] == 35, "One sealed35-node capture required")
    header = e.json(ledger["request_ref"])
    trace_ref = reference(directory / "actual_sigma_trace.json")
    trace = e.json(trace_ref)
    require(trace["complete"] is True and len(trace["steps"]) == 35, "Complete actual request trace required")
    base, new = policy.baseline, policy.terminal
    heads = ((base.coefficients, base.center, base.scale), (new.coefficients, new.center, new.scale))
    _, changed = compact_points(e, ledger, header, directory, trace, head=heads)
    for relative in (
        "scripts/verify_wam_terminal_points.py",
        "src/s2a/terminal_candidate.py",
        "src/s2a/terminal_capture_identity.py",
    ):
        ref = reference(Path(relative))
        require(
            p["source_sha256"].get(ref["path"]) == ref["sha256"], "Witness dependency differs from frozen execution"
        )
        e.path(ref)
    e.finish()
    publish(
        root / "first_candidate_numerical_witness.json",
        dict(
            passed=True,
            complete=True,
            partial=True,
            prepared_ref=prepared_ref,
            params_ref=params_ref,
            request_name=row["name"],
            ledger_ref=ledger_ref,
            trace_ref=trace_ref,
            nodes_verified=35,
            changed_active_values=changed,
            input_sha256=e.inputs,
            source_ref=reference(Path(__file__)),
            numerical_contract_passed=False,
            lifecycle_passed=False,
            performance_qualified=False,
            new_model_calls=0,
            new_fits=0,
            physics_steps=0,
            fresh_cases=0,
            boundary="First sealed request only; full99/3465 and real normal exit/release remain required before future state scoring.",
        ),
    )


if __name__ == "__main__":
    main()
