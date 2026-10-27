"""Read-only joint048 terminal qualification for the subsequent development stage."""

import argparse
import os
from pathlib import Path

from run_wam_clean_training_state_handoff import check_worker, worker_argv

from s2a.clean_training_d1_admission import Evidence
from s2a.clean_training_state_handoff import PREFIX, ROOT, admit
from s2a.durable_stage_supervisor import REPO, check_cpu_environment, reference, require
from s2a.owned_receipt_contract import normal_receipts


def qualify(registration_ref):
    require(Path.cwd().resolve() == REPO, "STATE terminal check requires S2A cwd")
    check_cpu_environment(os.environ)
    a = admit(registration_ref, role="owner")
    e = Evidence()
    result_ref = reference(ROOT / "supervisor/result.json")
    result = e.json(result_ref)
    require(
        result["schema_version"] == PREFIX + "owner.v1"
        and result["complete"] is result["passed"] is True
        and result["registration_ref"] == registration_ref
        and result["stage"] == "STATE"
        and result["w2_owner_ref"] == a["w2_owner_ref"]
        and result["w2_params_ref"] == a["w2_params_ref"]
        and result["automatic_retry"] is False
        and result["development_authorized"] is result["fresh_performance_authorized"] is False
        and result["numerical_contract_passed"] is result["eligible_for_development"] is True,
        "complete joint048 terminal qualification required",
    )
    require(
        result["owned_wrapper_ref"] == reference(ROOT / "supervisor/STATE/result.json"),
        "048 wrapper owner path differs",
    )
    owned = e.json(result["owned_wrapper_ref"])
    require(
        owned["complete"] is owned["passed"] is True and owned["timed_out"] is owned["automatic_retry"] is False,
        "048 wrapper did not finish normally",
    )
    launch = normal_receipts(e, owned["launch_ref"], owned["exit_ref"], owned["release_ref"], format="durable")
    require(launch["command"] == worker_argv(registration_ref), "048 actual wrapper command differs")
    worker_ref = check_worker(a, registration_ref)
    require(result["worker_ref"] == worker_ref, "048 terminal worker differs")
    worker = e.json(worker_ref)
    require(
        result["state_owner_ref"] == worker["delegate_result_ref"]
        and result["verification_ref"] == worker["verification_ref"]
        and type(result["state_gate_passed"]) is bool
        and result["state_gate_passed"] == worker["state_gate_passed"]
        and all(worker["identity"][k] == launch["child"][k] for k in ("pid", "start_ticks", "boot_id")),
        "048 worker identity, STATE audit or diagnostic gate differs",
    )
    e.mapping(a["source_sha256"])
    e.mapping(a["snapshot_sha256"])
    e.finish()
    return dict(
        registration_ref=registration_ref,
        owner_ref=result_ref,
        state_owner_ref=result["state_owner_ref"],
        verification_ref=result["verification_ref"],
        state_gate_passed=result["state_gate_passed"],
        numerical_contract_passed=True,
        eligible_for_development=True,
    )


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(dict(path=str(args.registration.resolve()), sha256=args.registration_sha256))))


if __name__ == "__main__":
    main()
