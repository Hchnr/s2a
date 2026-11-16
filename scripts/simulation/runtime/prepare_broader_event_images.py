"""Bind broader event-image replay to fixed original commands and endpoints."""

from pathlib import Path

import numpy as np

from scripts.simulation.runtime import prepare_broader_recovery_pilot as pilot
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_recovery_event_images as previous_images
from scripts.simulation.runtime import run_reset_correct_reference as recorded

ROOT = prefix.REPO
CONFIG = ROOT / "configs/wam/training/recovery/broader_event_images_20261116.json"
PLAN = ROOT / "plans/v0.0.1/wam/training/recovery/broader_event_images_20261116.md"
OUTPUT = ROOT / "outputs/broader_event_images_20261116"
ANCHORS = ((89, 118),) * 4 + ((90, 121),) * 4 + ((136, 202),) * 4 + ((167,),) * 4 + ((145,),) * 4 + ((135,),) * 4
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
    "event_audit": (
        "outputs/broader_recovery_events_20261116/result.json",
        "8d3b4878a7e3ecf2a597e5e8f5296deed7e624422816df871de96327b5d9e48b",
    ),
    "event_independent": (
        "outputs/broader_recovery_events_20261116/independent.json",
        "9a98f535fb3524e55e4d0fa3bae10cae284863d447d900f4a63b4a0266d20ee3",
    ),
    "event_union": (
        "outputs/broader_recovery_events_20261116/union.json",
        "b46077c35855bd7785cbb0444d6768e50cec98b0f88541a226032fce773bee60",
    ),
    "normalization": (
        "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/capture/D0/normalization.json",
        "9eb7fb8b9885958d8b947fd20727ae301813df068ec3dcaf1c3d8865180fedd9",
    ),
}


def validate_config(config):
    fixed = dict(
        schema="s2a.broader-event-images.v1",
        output="outputs/broader_event_images_20261116",
        case_indices=list(range(24)),
        anchors_by_case=[list(anchors) for anchors in ANCHORS],
        horizon=40,
        seed=42,
        probe_cases=[0],
        max_probe_control_periods=0,
        max_probe_inner_steps=0,
        max_probe_models=3,
        max_probe_contexts=3,
        max_workers=24,
        max_replay_control_periods=3552,
        max_replay_inner_steps=88800,
        max_replay_models=72,
        max_replay_contexts=72,
        max_saved_event_images=36,
        max_unsaved_camera_reads=456,
        worker_timeout_seconds=3600,
        world_model_calls=0,
        gpu_calls=0,
        fits=0,
        fresh_cases=0,
        training_authorized=False,
        automatic_retry=False,
        **{key: dict(path=path, sha256=sha) for key, (path, sha) in REFERENCES.items()},
    )
    prefix.require(pilot._same_typed(config, fixed), "The fixed event-image roster, sources and budgets are required")


def source_refs():
    """Bind both the previous replay machinery and the current six-source controls."""
    return (
        previous_images.source_refs()
        + pilot.source_refs()
        + [
            prefix.reference(path)
            for path in (
                CONFIG,
                PLAN,
                Path(__file__).resolve(),
                ROOT / "tests/research/native_runtime/test_broader_event_images_preparation.py",
            )
        ]
    )


def bind_case(index, row, event, source, producer):
    """Reject changed source, candidate or command provenance before replay."""
    prefix.require(
        row["case"] == source == producer["case"]
        and source["index"] == event["case_index"] == index
        and row["qualified"] is row["replayed"] is row["evidence_complete"] is True
        and event["source_identity"]["initial_state_sha256"] == source["initial_state_sha256"]
        and event["source_identity"]["task_index"] == source["source"]["task_index"]
        and event["source_identity"]["demo"] == source["source"]["demo"]
        and event["trajectory_ref"] == producer["arrays_ref"]
        and producer["complete"] is producer["passed"] is True
        and [item["anchor_step"] for item in event["candidates"]] == list(ANCHORS[index])
        and all(
            item["current_image_available"] is False and item["existing_image_ref"] is None
            for item in event["candidates"]
        ),
        "Each offline candidate must retain its original replayed training source and missing-image identity",
    )
    for item in event["candidates"]:
        step = item["anchor_step"]
        prefix.require(
            step + 40 <= source["periods"]
            and item["current_endpoint"] == producer["endpoint_receipts"][step]
            and item["selection_uses_future_training_labels"] is True
            and item["full_h40"] is True,
            "Each offline candidate needs its actual endpoint and completed original H40",
        )
    return dict(
        index=index,
        source=source,
        producer_ref=row["producer_ref"],
        checker_ref=row["checker_ref"],
        arrays_ref=producer["arrays_ref"],
        anchors=list(ANCHORS[index]),
        replay_periods=max(ANCHORS[index]),
        candidates=event["candidates"],
        pointwise_exception_case=index == 13,
    )


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    refs = {key: preparation.configured_ref(config[key]) for key in REFERENCES}
    recovery, h40, h40_check, events, event_check, union, _normalization = (
        prefix.read(refs[key]) for key in REFERENCES
    )
    _pilot_config, original = pilot.load_inputs()
    prefix.require(
        all(
            item["complete"] is item["passed"] is True
            for item in (recovery, h40, h40_check, events, event_check, union)
        )
        and recovery["registered_cases"]
        == len(recovery["cases"])
        == h40["registered_cases"]
        == len(events["cases"])
        == 24
        and h40["window_count"] == 72
        and h40["normalization_ref"] == refs["normalization"]
        and h40["validation_identity"]["identity_sets_disjoint"] is True
        and h40_check["primary_ref"] == refs["h40_audit"]
        and event_check["primary_ref"] == union["primary_ref"] == refs["event_audit"]
        and union["independent_ref"] == refs["event_independent"]
        and events["candidate_count"] == events["candidates_needing_current_image"] == 36
        and events["proposed_replay_control_periods"] == 3552
        and events["proposed_replay_inner_steps"] == 88800
        and events["proposed_unsaved_camera_reads"] == 456
        and all(
            item["training_authorized"] is False for item in (recovery, h40, h40_check, events, event_check, union)
        ),
        "The complete replayed control, H40 and independent event evidence chains are required",
    )
    cases = []
    for index in range(24):
        row, event, source = recovery["cases"][index], events["cases"][index], original["cases"][index]
        producer = prefix.read(row["producer_ref"])
        prefix.checked(row["checker_ref"])
        case = bind_case(index, row, event, source, producer)
        with np.load(prefix.checked(producer["arrays_ref"]), allow_pickle=False) as archive:
            commands = np.asarray(archive["sent_commands"][: case["replay_periods"]], dtype=np.float64)
            actual = np.asarray(
                [item["command"] for item in recorded.periods(producer)[: case["replay_periods"]]], dtype=np.float64
            )
            prefix.require(
                commands.shape == (case["replay_periods"], 7) and np.array_equal(commands, actual),
                "Only the actual dispatched original controller commands may be replayed",
            )
        cases.append(case)
    prefix.require(
        sum(case["replay_periods"] for case in cases) == config["max_replay_control_periods"]
        and sum(len(case["anchors"]) for case in cases) == config["max_saved_event_images"]
        and sum((case["replay_periods"] + 7) // 8 for case in cases) == config["max_unsaved_camera_reads"]
        and cases[13]["replay_periods"] == 167 < 295,
        "The complete physical and image budgets, including the exception boundary, must match",
    )
    return config, dict(config_ref=config_ref, refs=refs, shared_receipt=original["shared_receipt"], cases=cases)
