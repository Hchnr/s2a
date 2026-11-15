"""Analyze actual recovery events and synchronized validation identity metadata."""

import hashlib
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.research.wam.training.recovery_event_coverage import event_coverage, validation_identity_check
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.verification.wam.training.audit_recovery_h40_contract import Evidence, load_data

ROOT = prefix.REPO
CONFIG = ROOT / "configs/wam/training/recovery/recovery_event_coverage_20261116.json"
PLAN = ROOT / "plans/v0.0.1/wam/training/recovery/recovery_event_coverage_20261116.md"
OUTPUT = ROOT / "outputs/recovery_event_coverage_20261116/result.json"
REFERENCES = {
    "h40_audit": (
        "outputs/recovery_h40_contract_20261116/result.json",
        "0753c4c30f432e6b09e103152238cb4f6f47dffa30ce76e60636e59f2329dcec",
    ),
    "recovery_audit": (
        "outputs/current_controller_recovery_repair_20261116/audit.json",
        "a10dc09801d90efd92933e536d04a1e71ea1da75b724105dc384205c4d61c7f6",
    ),
    "training_preparation": (
        "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/preparation/prepared.json",
        "e6fab51219fba6d108bd6b858f2467a9a0e54c5fc22890f8abd7c318742f8e54",
    ),
    "validation_preflight": (
        "outputs/synchronous_validation_preparation_20261024/source_identity_preflight.json",
        "c435cd2c9da5521caea2a0916033312a17a0f9e53351814c16e6759e5a7b0d3d",
    ),
    "validation_prepared": (
        "outputs/synchronous_validation_20261024/prepared.json",
        "ba343c86753e56d20364e20f776d7983d6d475494753dbdfbead7cf9e75f23ad",
    ),
    "validation_verified": (
        "outputs/synchronous_validation_20261024/verified.json",
        "abe0a9982fa54c6a8425c90933d60c19fa667d129ee87a2707e6ddf1bf50a6d8",
    ),
}


def validate_config(config):
    expected = dict(
        schema="s2a.recovery-event-coverage.v1",
        output="outputs/recovery_event_coverage_20261116",
        case_indices=list(range(9)),
        existing_anchors=[0, 16, 56],
        opening_below=0.35,
        event_lead_periods=8,
        horizon=40,
        seed=42,
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
        fits=0,
        fresh_cases=0,
        new_current_images=0,
        training_authorized=False,
        automatic_retry=False,
    )
    prefix.require(
        isinstance(config, dict)
        and set(config) == set(expected) | set(REFERENCES)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in expected.items()),
        "The registered event definition, roster and zero-execution budget must match",
    )
    for key, (path, sha) in REFERENCES.items():
        prefix.require(config[key] == dict(path=path, sha256=sha), "The registered evidence identity must match")


def audit():
    prefix.environment()
    original.require_no_simulator_import()
    evidence = Evidence()
    config_ref = prefix.reference(CONFIG)
    config = evidence.json(config_ref)
    validate_config(config)
    for path in (
        PLAN,
        Path(__file__).resolve(),
        ROOT / "src/s2a/research/wam/training/recovery_event_coverage.py",
        ROOT / "src/s2a/research/wam/training/recovery_h40_contract.py",
        ROOT / "scripts/verification/wam/training/audit_recovery_h40_contract.py",
        ROOT / "tests/research/wam/training/test_recovery_event_coverage.py",
    ):
        evidence.path(prefix.reference(path))
    refs = {key: preparation.configured_ref(config[key]) for key in REFERENCES}
    h40, recovery, training, preflight, prepared, verification = (evidence.json(refs[key]) for key in REFERENCES)
    prefix.require(
        h40["complete"] is h40["passed"] is recovery["complete"] is recovery["passed"] is True
        and h40["window_count"] == 27
        and h40["unique_case_transition_rows"] == 864
        and h40["training_authorized"] is False
        and recovery["registered_cases"] == len(recovery["cases"]) == 9
        and preflight["prepared_ref"] == refs["training_preparation"]
        and prepared["preflight_ref"] == refs["validation_preflight"]
        and verification["prepared_ref"] == refs["validation_prepared"],
        "The original audits and later synchronized validation chain must match",
    )
    identity = validation_identity_check(h40["source_groups"], training, preflight, prepared, verification)
    cases = []
    for index, row in enumerate(recovery["cases"]):
        spec = row["case"]
        prefix.require(
            spec["index"] == index and row["qualified"] is row["replayed"] is row["evidence_complete"] is True,
            "Every original recovery case must have closed independent evidence",
        )
        group = next(group for group in h40["source_groups"] if index in group["case_indices"])
        prefix.require(
            spec["task_index"] == group["task_index"] and spec["initial_sha256"] == group["initial_state_sha256"],
            "The recovery case must remain in its original demo group",
        )
        branch = evidence.json(row["producer_ref"])
        prefix.require(branch["case"] == spec and branch["passed"] is True, "The original producer must match")
        data = load_data(evidence, branch["arrays_ref"])
        prefix.require(
            len(data["opening"]) == len(data["native_success"]) == len(data["sim_states"]) == spec["periods"] + 1
            and np.array_equal(data["endpoint_step"], np.arange(spec["periods"] + 1))
            and hashlib.sha256(np.ascontiguousarray(data["sim_states"][0]).tobytes()).hexdigest()
            == spec["initial_sha256"],
            "The full measured endpoint timeline and original reset must remain intact",
        )
        summary = event_coverage(
            data["opening"],
            data["native_success"],
            anchors=tuple(config["existing_anchors"]),
            threshold=config["opening_below"],
            lead=config["event_lead_periods"],
            horizon=config["horizon"],
        )
        candidate = summary["selected_anchor"]
        if candidate is not None:
            anchor = candidate["anchor_step"]
            candidate["sim_state_sha256"] = hashlib.sha256(
                np.ascontiguousarray(data["sim_states"][anchor]).tobytes()
            ).hexdigest()
            candidate["sim_time"] = float(data["sim_time"][anchor])
        cases.append(
            dict(
                case_index=index,
                task_index=spec["task_index"],
                source_demo=group["demo"],
                source_initial_sha256=group["initial_state_sha256"],
                offset_m=spec["offset_m"],
                trajectory_ref=branch["arrays_ref"],
                **summary,
            )
        )
    prefix.require(len(cases) == 9, "The full nine-case denominator is required")
    evidence.verify()
    original.require_no_simulator_import()
    return dict(
        complete=True,
        passed=True,
        config_ref=config_ref,
        source_refs=list(evidence.refs.values()),
        validation_identity=identity,
        old_state_initial_sync_certified=h40["original_validation_initial_sync_certified"],
        cases=cases,
        registered_cases=9,
        total_control_periods=sum(case["periods"] for case in cases),
        low_opening_total=sum(case["low_opening_total"] for case in cases),
        low_opening_covered=sum(case["low_opening_covered"] for case in cases),
        selected_anchor_count=sum(case["selected_anchor"] is not None for case in cases),
        training_authorized=False,
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
        fits=0,
        fresh_cases=0,
        new_current_images=0,
    )


def main():
    if OUTPUT.exists():
        raise SystemExit("Audit output already exists; no automatic retry or overwrite")
    try:
        result = audit()
    except Exception as error:
        publish_result(
            OUTPUT, dict(complete=False, passed=False, error=dict(type=type(error).__name__, message=str(error)))
        )
        raise
    print(publish_result(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()
