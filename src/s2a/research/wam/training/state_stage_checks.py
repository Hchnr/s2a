"""Read-only validation of the registered original STATE stage receipts."""

from pathlib import Path

from s2a.infrastructure.lifecycle.durable_stage_supervisor import REPO, read, reference, require
from s2a.research.wam.training.clean.clean_training_w1_handoff import PREPARED_ROOT
from s2a.research.wam.training.clean.clean_training_w2_admission import normal_receipts


def check_stage(a, ref):
    require(
        ref == reference(PREPARED_ROOT / "orchestration/STATE/result.json"), "048 original STATE owner path differs"
    )
    owner = read(ref)
    require(
        owner["schema_version"] == "s2a.clean-training-state-owner.v1"
        and owner["complete"] is owner["passed"] is True
        and owner["stage"] == "STATE"
        and owner["prepared_ref"] == a["prepared_ref"]
        and owner["execution_registration_ref"] == a["execution_registration_ref"]
        and owner["numerical_contract_passed"] is owner["eligible_for_development"] is True,
        "048 original STATE owner incomplete",
    )
    producer, audit = read(owner["producer_ref"]), read(owner["verification_ref"])
    for value, schema in (
        (producer, "s2a.clean-training-state-sampling.v1"),
        (audit, "s2a.clean-training-state-verification.v1"),
    ):
        require(
            value["schema_version"] == schema
            and value["complete"] is value["passed"] is True
            and value["prepared_ref"] == a["prepared_ref"]
            and value["execution_registration_ref"] == a["execution_registration_ref"]
            and value["actual_requests"] == 99
            and value["actual_denoise_calls"] == 3465
            and value["w2_fit_ref"] == a["w2_fit_ref"]
            and value["w2_verification_ref"] == a["w2_verification_ref"]
            and value["readout_ref"] == a["w2_params_ref"],
            "048 original STATE data/audit lineage or counts differ",
        )
    require(
        producer["model_initialization_attempts"]
        == producer["model_initializations"]
        == audit["model_initializations"]
        == 1
        and producer["actual_head_calls"] == 3465
        and all(producer[key] == 0 for key in ("future_label_reads", "producer_fits", "physics_steps", "fresh_cases"))
        and audit["physics_tasks_executed"] == audit["fresh_cases"] == 0
        and audit["zero_fullpoint_nodes_verified"] == 105
        and audit["actual_solver_sigma_traces_verified"] == 99,
        "048 original STATE model/online/zero/trace budget differs",
    )
    require(
        audit["sampling_ref"] == owner["producer_ref"]
        and audit["numerical_contract_passed"] is audit["eligible_for_development"] is True
        and type(audit["state_gate_passed"]) is bool
        and owner["state_gate_passed"] == audit["state_gate_passed"]
        and audit["paired_initial_conditions"] == 48
        and audit["zero_requests_verified"] == 3
        and audit["actual_head_calls"] == 3465,
        "048 full STATE numerical qualification differs",
    )
    require(
        len(owner["owned_commands"]) == 2
        and owner["owned_commands"][0]["launch"] != owner["owned_commands"][1]["launch"],
        "048 STATE lacks distinct model/audit owners",
    )
    prepared = read(a["prepared_ref"])
    from s2a.research.wam.training.clean.clean_training_d1_admission import Evidence

    e = Evidence()
    for index, command in enumerate(owner["owned_commands"]):
        launch = normal_receipts(e, command["launch"], command["exit"], command["release"], legacy=True)
        expected_python = prepared["model_config"]["model_python"] if index == 0 else str(REPO / ".venv/bin/python")
        expected_script = a["original_sampler_ref"] if index == 0 else a["original_verifier_ref"]
        require(
            launch["command"][0] == expected_python
            and Path(launch["command"][1]).resolve() == Path(expected_script["path"]),
            "048 original model/audit command differs",
        )
    gpu = read(owner["gpu_after_ref"])
    require(
        gpu["released"] is True
        and gpu["uuid"] == a["gpu_uuid"]
        and gpu["memory_mib"] == 0
        and gpu["compute_apps"] == []
        and audit["gpu_after_ref"] == owner["gpu_after_ref"],
        "048 GPU normal release missing",
    )
    e.finish()
    return owner
