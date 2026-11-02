"""Registered STATE worker command and closed-receipt checks, without launch."""

from s2a.infrastructure.lifecycle.durable_stage_supervisor import REPO, read, reference, require
from s2a.research.wam.training.clean.clean_training_state_handoff import PREFIX, ROOT

WORKER = "scripts/wam/training/clean/execute_wam_clean_training_state_handoff.py"


def worker_argv(ref):
    return [
        str(REPO / ".venv/bin/python"),
        WORKER,
        "--registration",
        ref["path"],
        "--registration-sha256",
        ref["sha256"],
        "--seed",
        "42",
    ]


def check_worker(a, ref):
    from s2a.research.wam.training.clean.clean_training_state_handoff import check_lookup_receipts
    from s2a.research.wam.training.state_stage_checks import check_stage

    result_ref = reference(ROOT / "worker/result.json")
    value = read(result_ref)
    require(
        value["schema_version"] == PREFIX + "worker.v1"
        and value["complete"] is value["passed"] is True
        and value["registration_ref"] == ref
        and value["w2_owner_ref"] == a["w2_owner_ref"]
        and value["w2_params_ref"] == a["w2_params_ref"]
        and value["delegate_calls_attempted"] == 1
        and value["source_before_after_exact"] is True
        and value["import_scope"]["restored"] is value["import_scope"]["source_guard_completed"] is True,
        "048 original STATE wrapper incomplete",
    )
    stage = check_stage(a, value["delegate_result_ref"])
    require(check_lookup_receipts(a, ref, stage) == value["lookup_refs"], "048 lookup refs differ")
    owner_lookup = read(value["lookup_refs"]["owner"])
    require(
        all(owner_lookup["identity"][k] == value["identity"][k] for k in ("pid", "start_ticks", "boot_id")),
        "048 owner lookup came from another wrapper",
    )
    require(
        value["producer_ref"] == stage["producer_ref"] and value["verification_ref"] == stage["verification_ref"],
        "048 stage output refs differ",
    )
    return result_ref
