"""Inspect measured training recovery events and offline image candidates."""

import hashlib
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.research.wam.training.broader_recovery_events import event_candidates
from s2a.research.wam.training.clean.clean_training_data import DOMAIN
from s2a.research.wam.training.recovery_h40_contract import audit_window, window_indices
from scripts.simulation.runtime import prepare_broader_recovery_pilot as pilot
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as feedback
from scripts.verification.wam.training import audit_broader_recovery_coverage as coverage
from scripts.verification.wam.training import audit_broader_recovery_h40 as h40_source
from scripts.verification.wam.training.audit_recovery_h40_contract import Evidence, load_data

ROOT = prefix.REPO
CONFIG = ROOT / "configs/wam/training/recovery/broader_recovery_events_20261116.json"
PLAN = ROOT / "plans/v0.0.1/wam/training/recovery/broader_recovery_events_20261116.md"
OUTPUT = ROOT / "outputs/broader_recovery_events_20261116/result.json"
ANCHORS = (0, 16, 56)
REFERENCES = {
    "recovery_audit": (
        "outputs/broader_recovery_pilot_20261116/audit.json",
        "16d8c1250690b540b3d7230df329ec4b00aee23770033cfa3ded0c68fcb1847f",
    ),
    "h40_audit": (
        "outputs/broader_recovery_h40_20261116/result.json",
        "ca4c9d8910be0949ed877072bcfe8f03d598760108d4ff1ee57f035565625ba4",
    ),
    "h40_independent": (
        "outputs/broader_recovery_h40_20261116/independent.json",
        "b709aecdc9afdaf36991d03af3b29ed312d6d6f6c99b2e5eccecd7c0e36edd5a",
    ),
    "previous_event_audit": (
        "outputs/recovery_event_coverage_20261116/result.json",
        "0618788ddbde31d9128f0b8de5b1daeb0dc838a88d7fdab81e61d0b3561aff29",
    ),
    "training_preparation": (
        "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/preparation/prepared.json",
        "e6fab51219fba6d108bd6b858f2467a9a0e54c5fc22890f8abd7c318742f8e54",
    ),
    "normalization": (
        "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/capture/D0/normalization.json",
        "9eb7fb8b9885958d8b947fd20727ae301813df068ec3dcaf1c3d8865180fedd9",
    ),
}


def validate_config(config):
    fixed = dict(
        schema="s2a.broader-recovery-events.v1",
        output="outputs/broader_recovery_events_20261116",
        case_indices=list(range(24)),
        existing_anchors=list(ANCHORS),
        opening_below=0.35,
        event_lead_periods=8,
        horizon=40,
        seed=42,
        max_candidates=48,
        max_union_case_transition_rows=4224,
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
        new_current_images=0,
        fits=0,
        fresh_cases=0,
        training_authorized=False,
        automatic_retry=False,
        **{key: dict(path=path, sha256=sha) for key, (path, sha) in REFERENCES.items()},
    )
    prefix.require(
        pilot._same_typed(config, fixed), "The fixed event roster, identities and no-execution scope are required"
    )


def candidate_details(evidence, data, producer, checker, identity, tool, normalization, candidate):
    """Bind a proposed endpoint and its already-executed future, without rendering."""
    step = candidate["anchor_step"]
    receipt = producer["endpoint_receipts"][step]
    prefix.require(
        receipt == checker["endpoint_receipts"][step]
        and receipt["endpoint_step"] == step
        and receipt["sim_state_sha256"]
        == hashlib.sha256(np.ascontiguousarray(data["sim_states"][step]).tobytes()).hexdigest()
        and receipt["sim_time"] == float(data["sim_time"][step]),
        "Each candidate must bind its actual independently replayed endpoint",
    )
    image_ref = None
    if candidate["current_image_available"]:
        matches = [image for image in producer["images"] if image["frame_index"] == step]
        prefix.require(
            len(matches) == 1 and matches[0]["current_endpoint"] == receipt, "The existing current image must match"
        )
        image_ref = matches[0]["canvas_png"]
        evidence.path(image_ref)
    labels = audit_window(data, step, tool, normalization)
    opening, command = labels.pop("observed_openings"), labels.pop("actual_gripper_commands")
    return dict(
        **candidate,
        source_identity=identity,
        current_endpoint=receipt,
        existing_image_ref=image_ref,
        opening_range=[float(opening.min()), float(opening.max())],
        opening_below_count=int(np.count_nonzero(opening < 0.35)),
        command_counts=h40_source.command_counts(command),
        **labels,
    )


def audit():
    prefix.environment()
    feedback.require_no_simulator_import()
    evidence = Evidence()
    config_ref = prefix.reference(CONFIG)
    config = evidence.json(config_ref)
    validate_config(config)
    for path in (
        PLAN,
        Path(__file__).resolve(),
        ROOT / "src/s2a/research/wam/training/broader_recovery_events.py",
        ROOT / "src/s2a/research/wam/training/recovery_h40_contract.py",
        ROOT / "src/s2a/research/wam/training/clean/clean_training_data.py",
        ROOT / "scripts/verification/wam/training/audit_broader_recovery_h40.py",
        ROOT / "tests/research/wam/training/test_broader_recovery_events.py",
        ROOT / "tests/research/wam/training/test_broader_recovery_events_audit.py",
    ):
        evidence.path(prefix.reference(path))
    refs = {key: preparation.configured_ref(config[key]) for key in REFERENCES}
    recovery, h40, independent, previous, prepared, normalization = (evidence.json(refs[key]) for key in REFERENCES)
    prefix.require(
        recovery["complete"] is recovery["passed"] is recovery["sources_unchanged"] is True
        and recovery["registered_cases"] == len(recovery["cases"]) == 24
        and h40["complete"] is h40["passed"] is independent["complete"] is independent["passed"] is True
        and independent["primary_ref"] == refs["h40_audit"]
        and h40["registered_cases"] == 24
        and h40["window_count"] == 72
        and h40["normalization_ref"] == refs["normalization"]
        and h40["validation_identity"]["identity_sets_disjoint"] is True
        and previous["complete"] is previous["passed"] is True
        and normalization["domain"] == DOMAIN
        and normalization["normalization_id"] == 31
        and normalization["projection_id"] == 0
        and recovery["training_authorized"]
        is h40["training_authorized"]
        is independent["training_authorized"]
        is False,
        "The replayed sources, H40 contract, validation separation and normalization must match",
    )
    cases, candidate_flags, union_rows, candidate_rows = [], np.zeros(10, dtype=np.int64), set(), set()
    missing_images = proposed_periods = unsaved_renders = 0
    for index, row in enumerate(recovery["cases"]):
        spec = row["case"]
        identity, original_anchor = h40_source.training_source(spec, prepared)
        prefix.require(h40["cases"][index]["source_identity"] == identity, "The audited training source must match")
        producer, checker = (evidence.json(row[key]) for key in ("producer_ref", "checker_ref"))
        tool_ref = spec["original_identity"]["refs"]["tool"]
        coverage.verify_pair(index, row, producer, checker, tool_ref)
        data = load_data(evidence, producer["arrays_ref"])
        evidence.path(checker["arrays_ref"])
        prefix.require(
            len(data["sim_states"]) == len(data["opening"]) == len(data["native_success"]) == spec["periods"] + 1
            and np.array_equal(data["endpoint_step"], np.arange(spec["periods"] + 1))
            and hashlib.sha256(np.ascontiguousarray(data["sim_states"][0]).tobytes()).hexdigest()
            == identity["initial_state_sha256"],
            "The complete original actual endpoint timeline must match",
        )
        tool = np.asarray(original_anchor["tool_source_from_tcp"], dtype=np.float64)
        np.testing.assert_allclose(tool, evidence.json(tool_ref)["source_from_tcp"], atol=1e-12, rtol=0)
        events = event_candidates(
            data["opening"],
            data["native_success"],
            anchors=ANCHORS,
            threshold=config["opening_below"],
            lead=config["event_lead_periods"],
            horizon=config["horizon"],
        )
        detailed = [
            candidate_details(evidence, data, producer, checker, identity, tool, normalization, candidate)
            for candidate in events.pop("candidates")
        ]
        candidate_indices = {
            value for item in detailed for value in range(item["anchor_step"], item["anchor_step"] + 40)
        }
        existing_indices = {
            value for window in window_indices(ANCHORS, 40, spec["periods"])["windows"] for value in window
        }
        candidate_rows.update((index, step) for step in candidate_indices)
        union_rows.update((index, step) for step in candidate_indices | existing_indices)
        candidate_flags += sum(
            (np.asarray(item["clipped_per_active_slot"], dtype=np.int64) for item in detailed),
            start=np.zeros(10, dtype=np.int64),
        )
        missing = [item["anchor_step"] for item in detailed if not item["current_image_available"]]
        periods = max(missing, default=0)
        missing_images += len(missing)
        proposed_periods += periods
        unsaved_renders += (periods + 7) // 8
        cases.append(
            dict(
                case_index=index,
                source_identity=identity,
                variant=spec["variant"],
                pointwise_exception_case=index == 13,
                producer_ref=row["producer_ref"],
                checker_ref=row["checker_ref"],
                trajectory_ref=producer["arrays_ref"],
                candidates=detailed,
                candidate_unique_transition_rows=len(candidate_indices),
                union_unique_transition_rows=len(candidate_indices | existing_indices),
                additional_unique_transition_rows=len(candidate_indices - existing_indices),
                proposed_replay_periods=periods,
                proposed_unsaved_camera_reads=(periods + 7) // 8,
                **events,
            )
        )
    count = sum(len(case["candidates"]) for case in cases)
    prefix.require(
        len(cases) == 24
        and count <= config["max_candidates"]
        and len(union_rows) <= config["max_union_case_transition_rows"]
        and len(
            {
                (
                    row["source_identity"]["task_index"],
                    row["source_identity"]["demo"],
                    row["source_identity"]["initial_state_sha256"],
                )
                for row in cases
            }
        )
        == 6,
        "The original case, source and candidate budgets must remain fixed",
    )
    evidence.verify()
    feedback.require_no_simulator_import()
    return dict(
        complete=True,
        passed=True,
        config_ref=config_ref,
        input_refs=list(evidence.refs.values()),
        cases=cases,
        registered_cases=len(cases),
        source_groups=6,
        total_control_periods=sum(case["periods"] for case in cases),
        low_opening_total=sum(case["low_opening"]["total"] for case in cases),
        low_opening_covered=sum(case["low_opening"]["covered"] for case in cases),
        native_success_total=sum(case["native_success"]["total"] for case in cases),
        native_success_covered=sum(case["native_success"]["covered"] for case in cases),
        candidate_count=count,
        candidates_needing_current_image=missing_images,
        candidate_window_label_rows=count * 40,
        candidate_unique_case_transition_rows=len(candidate_rows),
        existing_and_candidate_unique_case_transition_rows=len(union_rows),
        candidate_clipping_per_active_slot=candidate_flags.tolist(),
        candidate_clipping_total=int(candidate_flags.sum()),
        proposed_replay_control_periods=proposed_periods,
        proposed_replay_inner_steps=25 * proposed_periods,
        proposed_unsaved_camera_reads=unsaved_renders,
        candidate_selection_uses_future_training_labels=True,
        training_authorized=False,
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
        new_current_images=0,
        fits=0,
        fresh_cases=0,
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
