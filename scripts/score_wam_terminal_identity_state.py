"""Read053 future endpoints only after terminal058 numerical/lifecycle qualification."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from verify_wam_clean_training_state import integrate

from s2a.clean_training_runtime import publish, read, reference, require
from s2a.terminal_identity_state_metrics import diagnosis


def score(prepared_ref, numerical_ref, output):
    numerical = read(numerical_ref)
    require(
        numerical["schema_version"] == "s2a.terminal-identity-state-audit.v1"
        and numerical["passed"]
        is numerical["complete"]
        is numerical["numerical_contract_passed"]
        is numerical["lifecycle_passed"]
        is True
        and numerical["prepared_ref"] == prepared_ref,
        "Full terminal058 numerical and normal lifecycle gate must precede any validation future read",
    )
    prepared = read(prepared_ref)
    require(
        prepared["schema_version"] == "s2a.terminal-identity-state-preparation.v1", "Terminal058 preparation required"
    )
    current = read(prepared["current_inputs_ref"])
    audit = read(current["source_audit_ref"])
    require(
        audit["passed"] is audit["eligible_for_model_validation"] is True and audit["native_stable_cases"] == 6,
        "Qualified053 native controls required",
    )
    require(len(numerical["generated_refs"]) == 96, "Full054 generated denominator")
    anchors = {r["name"]: r for r in current["current_inputs"]}
    archives, archive_refs, rows = {}, {}, []
    for item in numerical["generated_refs"]:
        anchor = anchors[item["anchor_name"]]
        matches = [
            (case, refs)
            for case, refs in zip(audit["cases"], audit["case_refs"], strict=True)
            if anchor["name"].startswith(case["name"] + "_a")
        ]
        require(len(matches) == 1, "Current/future case identity is ambiguous")
        case, refs = matches[0]
        name = case["name"]
        if name not in archives:
            path = Path(refs["arrays_ref"]["path"])
            require(reference(path) == refs["arrays_ref"], "053 future archive SHA changed")
            with np.load(path, allow_pickle=False) as z:
                archives[name] = {k: z[k].copy() for k in ["sim_time", "sim_states", "state128", "action_valid"]}
            archive_refs[name] = refs["arrays_ref"]
        data = archives[name]
        frame = anchor["anchor_frame"]
        require(
            data["sim_time"][frame] == anchor["current_time_s"]
            and hashlib.sha256(data["sim_states"][frame].tobytes()).hexdigest() == anchor["current_sim_state_sha256"]
            and np.array_equal(data["state128"][frame], anchor["measured_state128"]),
            "Current/future physical binding differs",
        )
        require(
            np.allclose(np.diff(data["sim_time"][frame : frame + 41]), 0.05, rtol=0, atol=1e-9)
            and data["action_valid"][frame : frame + 40].all(),
            "Incomplete real H40/20Hz labels",
        )
        response = read(item["response_ref"])
        action = np.asarray(response["outputs"][0]["content"]["action"])
        errors = integrate(data["state128"], anchor, action)
        rows.append(
            {
                "name": item["name"],
                "episode": name,
                "task_index": int(name[4]),
                "sampling_seed": item["sampling_seed"],
                "variant": item["variant"],
                "errors": errors,
            }
        )
    groups, gate = diagnosis(rows)
    for ref in archive_refs.values():
        require(reference(ref["path"]) == ref, "053 labels changed while scoring")
    result = {
        "schema_version": "s2a.terminal-identity-state-score.v1",
        "complete": True,
        "passed": True,
        "prepared_ref": prepared_ref,
        "candidate_proof_ref": prepared["candidate_proof_ref"],
        "candidate_params_ref": prepared["candidate_params_ref"],
        "numerical_ref": numerical_ref,
        "state_metrics": groups,
        "state_gate": gate,
        "state_gate_passed": gate["passed"],
        "eligible_for_development": gate["passed"],
        "state_rows": rows,
        "future_cases_read": len(archives),
        "future_archive_refs": archive_refs,
        "new_model_calls": 0,
        "physics_steps": 0,
        "fresh_cases": 0,
        "evidence_boundary": "Synchronized generated-reference proximity; not robot tracking or task success",
        "fresh_authorized": False,
    }
    return publish(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--numerical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(reference(args.prepared), reference(args.numerical), args.output)), flush=True)


if __name__ == "__main__":
    main()
