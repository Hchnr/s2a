"""Audit the 27 measured recovery H40 windows without running or fitting a model."""

import hashlib
import json
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.research.wam.training.clean.clean_training_data import DOMAIN
from s2a.research.wam.training.recovery_h40_contract import audit_window, source_identity, window_indices
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.verification.wam.training.verify_settled_native_training import verify_canvas

ROOT = prefix.REPO
CONFIG = ROOT / "configs/wam/training/recovery/recovery_h40_contract_20261116.json"
PLAN = ROOT / "plans/v0.0.1/wam/training/recovery/recovery_h40_contract_20261116.md"
OUTPUT = ROOT / "outputs/recovery_h40_contract_20261116/result.json"
REFERENCES = {
    "recovery_audit": (
        "outputs/current_controller_recovery_repair_20261116/audit.json",
        "a10dc09801d90efd92933e536d04a1e71ea1da75b724105dc384205c4d61c7f6",
    ),
    "training_preparation": (
        "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/preparation/prepared.json",
        "e6fab51219fba6d108bd6b858f2467a9a0e54c5fc22890f8abd7c318742f8e54",
    ),
    "normalization": (
        "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/capture/D0/normalization.json",
        "9eb7fb8b9885958d8b947fd20727ae301813df068ec3dcaf1c3d8865180fedd9",
    ),
    "d0_capture": (
        "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/capture/D0/result.json",
        "d23b31eb9e40f9f3e8f22571399d164d8980ba3f4d75b10876d740edd52b7dae",
    ),
}
SOURCES = ((0, "demo_8"), (1, "demo_1"), (2, "demo_2"))
OFFSETS = (0.0, 0.02, -0.02)
ANCHORS = (0, 16, 56)


def validate_config(config):
    expected = dict(
        schema="s2a.recovery-h40-contract-audit.v1",
        output="outputs/recovery_h40_contract_20261116",
        case_indices=list(range(9)),
        sources=[list(item) for item in SOURCES],
        offsets_m=list(OFFSETS),
        anchor_steps=list(ANCHORS),
        horizon=40,
        seed=42,
        control_dt_s=0.05,
        max_cases=9,
        max_windows=27,
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
        fits=0,
        fresh_cases=0,
        training_authorized=False,
        automatic_retry=False,
    )
    prefix.require(
        isinstance(config, dict)
        and set(config) == set(expected) | set(REFERENCES)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in expected.items()),
        "The exact audit roster, source groups and no-model budget are required",
    )
    for key, (path, sha) in REFERENCES.items():
        prefix.require(config[key] == dict(path=path, sha256=sha), "The registered input identity must remain fixed")


class Evidence:
    """Bind every consumed file and verify it again after analysis."""

    def __init__(self):
        self.refs = {}

    def path(self, ref):
        path = prefix.checked(ref)
        value = prefix.reference(path)
        prefix.require(value["path"] not in self.refs or self.refs[value["path"]] == value, "Conflicting source SHA")
        self.refs[value["path"]] = value
        return path

    def json(self, ref):
        return json.loads(self.path(ref).read_text())

    def verify(self):
        for ref in self.refs.values():
            prefix.checked(ref)


def load_data(evidence, ref):
    with np.load(evidence.path(ref), allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive.files}


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
        ROOT / "src/s2a/research/wam/training/recovery_h40_contract.py",
        ROOT / "src/s2a/research/wam/training/clean/clean_training_data.py",
        ROOT / "scripts/verification/wam/training/verify_settled_native_training.py",
        ROOT / "tests/research/wam/training/test_recovery_h40_contract.py",
    ):
        evidence.path(prefix.reference(path))
    refs = {key: preparation.configured_ref(value) for key, value in config.items() if key in REFERENCES}
    previous, prepared, normalization, d0 = (evidence.json(refs[key]) for key in REFERENCES)
    prefix.require(
        previous["complete"] is previous["passed"] is previous["sources_unchanged"] is True
        and previous["training_authorized"] is False
        and previous["zero_offset_qualified"] == 3
        and previous["nonzero_qualified"] == 6
        and previous["registered_cases"] == len(previous["cases"]) == 9
        and d0["complete"] is d0["passed"] is True
        and d0["prepared_ref"] == refs["training_preparation"]
        and d0["normalization_ref"] == refs["normalization"]
        and normalization["domain"] == DOMAIN
        and normalization["normalization_id"] == 31
        and normalization["projection_id"] == 0,
        "The complete recovery audit and original D0 training domain must match",
    )
    windows, groups, clipped, openings, commands, unique_rows = [], {}, np.zeros(10, dtype=np.int64), [], [], set()
    for index, row in enumerate(previous["cases"]):
        spec = row["case"]
        prefix.require(
            spec["index"] == index
            and spec["offset_m"] == OFFSETS[index % 3]
            and spec["seed"] == 42
            and row["qualified"] is row["replayed"] is row["evidence_complete"] is True,
            "The nine qualified, independently replayed cases must remain in order",
        )
        identity = source_identity(spec, prepared, SOURCES[index // 3])
        group = (identity["task_index"], identity["demo"], identity["initial_state_sha256"])
        if index % 3 == 0:
            prefix.require(group not in groups, "Each original demo must be a distinct group")
            groups[group] = []
        prefix.require(group in groups, "All variants must retain their original demo group")
        groups[group].append(index)
        producer, checker = (evidence.json(row[key]) for key in ("producer_ref", "checker_ref"))
        prefix.require(
            producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True
            and producer["case"] == checker["case"] == spec
            and producer["physical_labels_from_reference"] is False
            and producer["label_semantics"]
            == dict(
                action128_opening="next_measured_opening_not_controller_command",
                sent_commands="actual_seven_dimensional_controller_command",
            ),
            "Producer and independent checker identities and measured-label semantics must match",
        )
        data = load_data(evidence, producer["arrays_ref"])
        prefix.require(
            len(data["sim_states"]) == spec["periods"] + 1
            and hashlib.sha256(np.ascontiguousarray(data["sim_states"][0]).tobytes()).hexdigest()
            == spec["initial_sha256"]
            and [image["frame_index"] for image in producer["images"]] == list(ANCHORS),
            "The original reset and all actual image anchors must be present",
        )
        grouping = window_indices(ANCHORS, 40, spec["periods"])
        prefix.require(grouping["total"] == 120 and grouping["unique"] == 96, "The actual overlapping windows differ")
        unique_rows.update((index, frame) for row_indices in grouping["windows"] for frame in row_indices)
        training_anchor = next(
            item
            for item in prepared["training_anchors"]
            if item["source_identity"] == identity and item["anchor_frame"] == 0
        )
        tool = np.asarray(training_anchor["tool_source_from_tcp"], dtype=np.float64)
        for frame, image in zip(ANCHORS, producer["images"], strict=True):
            receipt = producer["endpoint_receipts"][frame]
            prefix.require(
                image["current_endpoint"] == receipt
                and receipt["endpoint_step"] == frame
                and receipt["sim_state_sha256"]
                == hashlib.sha256(np.ascontiguousarray(data["sim_states"][frame]).tobytes()).hexdigest()
                and receipt["sim_time"] == float(data["sim_time"][frame]),
                "Each actual image must bind its own measured endpoint and clock",
            )
            for key in ("raw_cameras", "canvas_png"):
                evidence.path(image[key])
            canvas_check = verify_canvas(image, data, receipt)
            result = audit_window(data, frame, tool, normalization)
            openings.append(result.pop("observed_openings"))
            commands.append(result.pop("actual_gripper_commands"))
            clipped += np.asarray(result["clipped_per_active_slot"], dtype=np.int64)
            windows.append(
                dict(
                    case_index=index,
                    source_identity=identity,
                    offset_m=spec["offset_m"],
                    anchor_step=frame,
                    trajectory_ref=producer["arrays_ref"],
                    image_ref=image["canvas_png"],
                    current_endpoint=receipt,
                    canvas_check=canvas_check,
                    **result,
                )
            )
    prefix.require(
        len(groups) == 3
        and all(len(indices) == 3 for indices in groups.values())
        and len({item[2] for item in groups}) == 3
        and len(windows) == 27
        and len(unique_rows) == 864,
        "Three source groups, 27 windows and 864 unique case/transition pairs are required",
    )
    opening, command = np.concatenate(openings), np.concatenate(commands)
    prefix.require(len(opening) == len(command) == 1080, "All 1,080 window rows must be reported")
    evidence.verify()
    original.require_no_simulator_import()
    return dict(
        complete=True,
        passed=True,
        source_refs=list(evidence.refs.values()),
        config_ref=config_ref,
        normalization_ref=refs["normalization"],
        original_validation_initial_sync_certified=prepared["exclusion"]["validation_initial_sync_certified"],
        unknown_near_duplicate_exclusion_claimed=False,
        source_groups=[
            dict(task_index=k[0], demo=k[1], initial_state_sha256=k[2], case_indices=v) for k, v in groups.items()
        ],
        registered_cases=9,
        windows=windows,
        window_count=len(windows),
        window_label_rows=len(opening),
        unique_case_transition_rows=len(unique_rows),
        repeated_window_label_rows=len(opening) - len(unique_rows),
        clipping_per_active_slot=clipped.tolist(),
        clipping_total=int(clipped.sum()),
        measured_opening_range=[float(opening.min()), float(opening.max())],
        sent_gripper_command_range=[float(command.min()), float(command.max())],
        sent_gripper_command_unique=np.unique(command).tolist(),
        training_authorized=False,
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
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
