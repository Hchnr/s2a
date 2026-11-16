"""Audit measured broader recovery windows against the fixed WAM data contract."""

import hashlib
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.research.wam.training.broader_recovery_h40 import clipping_flags, validate_six_sources
from s2a.research.wam.training.clean.clean_training_data import DOMAIN
from s2a.research.wam.training.recovery_h40_contract import audit_window, window_indices
from scripts.simulation.runtime import prepare_broader_recovery_pilot as pilot
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as feedback
from scripts.verification.wam.training import audit_broader_recovery_coverage as coverage
from scripts.verification.wam.training.audit_recovery_h40_contract import Evidence, load_data
from scripts.verification.wam.training.verify_settled_native_training import verify_canvas

ROOT = prefix.REPO
CONFIG = ROOT / "configs/wam/training/recovery/broader_recovery_h40_20261116.json"
PLAN = ROOT / "plans/v0.0.1/wam/training/recovery/broader_recovery_h40_20261116.md"
OUTPUT = ROOT / "outputs/broader_recovery_h40_20261116/result.json"
ANCHORS = (0, 16, 56)
REFERENCES = {
    "recovery_audit": (
        "outputs/broader_recovery_pilot_20261116/audit.json",
        "16d8c1250690b540b3d7230df329ec4b00aee23770033cfa3ded0c68fcb1847f",
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
    "coverage_audit": (
        "outputs/broader_recovery_coverage_20261116/result.json",
        "b4f3c19f922c5f886ed6bd3dc19c6d7e16aacb8936a090b09df475fda317f920",
    ),
    "coverage_independent": (
        "outputs/broader_recovery_coverage_20261116/independent.json",
        "ba4450000ef8732df9b67b38c253fc8e771111c353efda097ba1eecb6a629966",
    ),
}


def validate_config(config):
    fixed = dict(
        schema="s2a.broader-recovery-h40.v1",
        output="outputs/broader_recovery_h40_20261116",
        case_indices=list(range(24)),
        anchor_steps=list(ANCHORS),
        horizon=40,
        seed=42,
        control_dt_s=0.05,
        source_groups=6,
        registered_cases=24,
        window_count=72,
        window_label_rows=2880,
        unique_case_transition_rows=2304,
        pointwise_exception_cases=[13],
        opening_below=0.35,
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
        fits=0,
        fresh_cases=0,
        training_authorized=False,
        automatic_retry=False,
        **{key: dict(path=path, sha256=sha) for key, (path, sha) in REFERENCES.items()},
    )
    prefix.require(
        pilot._same_typed(config, fixed), "The fixed H40 roster, evidence and zero-training budget are required"
    )


def training_source(case, prepared):
    """Bind one case to an original training demo, excluding held-out metadata."""
    source = case["source"]
    identity = dict(
        task_index=source["task_index"], demo=source["demo"], initial_state_sha256=case["initial_state_sha256"]
    )
    rows = [
        row for row in prepared["training_anchors"] if row["source_identity"] == identity and row["anchor_frame"] == 0
    ]
    excluded = prepared["exclusion"]
    key = (identity["task_index"], identity["demo"])
    prefix.require(
        len(rows) == 1
        and key not in {tuple(row) for row in excluded["forbidden_demo_keys"] + excluded["validation_demo_keys"]}
        and identity["initial_state_sha256"] not in excluded["forbidden_initial_state_sha256"]
        and not any(
            (row["task_index"], row["demo"]) == key or row["initial_state_sha256"] == identity["initial_state_sha256"]
            for row in excluded["fresh_sources"]
        ),
        "One original training source outside validation and fresh identities is required",
    )
    return identity, rows[0]


def command_counts(values):
    values, counts = np.unique(np.asarray(values, dtype=np.float64), return_counts=True)
    return {str(0.0 if value == 0 else float(value)): int(count) for value, count in zip(values, counts, strict=True)}


def state_pair_hashes(states, indices):
    """Count exact simulator-state byte pairs, without claiming independence."""
    hashes = [hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest() for state in states]
    return {(hashes[index], hashes[index + 1]) for index in indices}


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
        ROOT / "src/s2a/research/wam/training/broader_recovery_h40.py",
        ROOT / "src/s2a/research/wam/training/recovery_h40_contract.py",
        ROOT / "src/s2a/research/wam/training/clean/clean_training_data.py",
        ROOT / "src/s2a/research/wam/training/recovery_event_coverage.py",
        ROOT / "scripts/verification/wam/training/audit_broader_recovery_coverage.py",
        ROOT / "scripts/verification/wam/training/verify_settled_native_training.py",
        ROOT / "tests/research/wam/training/test_broader_recovery_h40.py",
        ROOT / "tests/research/wam/training/test_broader_recovery_h40_audit.py",
    ):
        evidence.path(prefix.reference(path))
    refs = {key: preparation.configured_ref(config[key]) for key in REFERENCES}
    recovery, prepared, normalization, d0, preflight, validation, verified, cov, cov_check = (
        evidence.json(refs[key]) for key in REFERENCES
    )
    prefix.require(
        recovery["complete"] is recovery["passed"] is recovery["sources_unchanged"] is True
        and recovery["registered_cases"] == recovery["evidence_complete_cases"] == len(recovery["cases"]) == 24
        and recovery["training_authorized"] is False
        and d0["complete"] is d0["passed"] is True
        and d0["prepared_ref"] == refs["training_preparation"]
        and d0["normalization_ref"] == refs["normalization"]
        and normalization["domain"] == DOMAIN
        and normalization["normalization_id"] == 31
        and normalization["projection_id"] == 0
        and preflight["prepared_ref"] == refs["training_preparation"]
        and validation["preflight_ref"] == refs["validation_preflight"]
        and verified["prepared_ref"] == refs["validation_prepared"]
        and cov["complete"] is cov["passed"] is cov_check["complete"] is cov_check["passed"] is True
        and cov_check["primary_ref"] == refs["coverage_audit"]
        and cov["training_authorized"] is cov_check["training_authorized"] is False,
        "The physical, normalization, validation and coverage evidence chains must match",
    )
    windows, groups, cases, unique_rows, state_pairs = [], [], [], set(), set()
    window_flags, unique_flags = [], []
    window_openings, unique_openings, window_commands, unique_commands = [], [], [], []
    pointwise_exceptions = []
    for index, row in enumerate(recovery["cases"]):
        spec = row["case"]
        identity, original_anchor = training_source(spec, prepared)
        if index % len(pilot.VARIANTS) == 0:
            prefix.require(
                not any(all(group[key] == value for key, value in identity.items()) for group in groups),
                "Each original demo must be a distinct group",
            )
            groups.append(dict(**identity, case_indices=[]))
        group = groups[-1]
        prefix.require(
            all(group[key] == identity[key] for key in identity), "All four variants must share the same source group"
        )
        group["case_indices"].append(index)
        producer, checker = (evidence.json(row[key]) for key in ("producer_ref", "checker_ref"))
        tool_ref = spec["original_identity"]["refs"]["tool"]
        coverage.verify_pair(index, row, producer, checker, tool_ref)
        tool = np.asarray(original_anchor["tool_source_from_tcp"], dtype=np.float64)
        np.testing.assert_allclose(tool, evidence.json(tool_ref)["source_from_tcp"], atol=1e-12, rtol=0)
        prefix.require(
            producer["label_semantics"]
            == checker["label_semantics"]
            == dict(
                action128_opening="next_measured_opening_not_controller_command",
                sent_commands="actual_seven_dimensional_controller_command",
            ),
            "Measured opening labels and sent commands must remain distinct",
        )
        data = load_data(evidence, producer["arrays_ref"])
        replay = load_data(evidence, checker["arrays_ref"])
        prefix.require(set(data) == set(replay), "Producer and checker actual trajectory schemas must match")
        for key in data:
            np.testing.assert_allclose(data[key], replay[key], atol=1e-10, rtol=0)
        prefix.require(
            len(data["sim_states"]) == spec["periods"] + 1
            and hashlib.sha256(np.ascontiguousarray(data["sim_states"][0]).tobytes()).hexdigest()
            == identity["initial_state_sha256"]
            and [image["frame_index"] for image in producer["images"]] == list(ANCHORS),
            "The initial state and all registered current images must match",
        )
        grouping = window_indices(ANCHORS, 40, spec["periods"])
        prefix.require(grouping["total"] == 120 and grouping["unique"] == 96, "The fixed actual windows must match")
        indices = sorted({frame for group_indices in grouping["windows"] for frame in group_indices})
        unique_rows.update((index, frame) for frame in indices)
        state_pairs.update(state_pair_hashes(data["sim_states"], indices))
        unique_clipping = clipping_flags(data["action128"][indices], normalization)
        unique_flags.append(unique_clipping)
        unique_openings.append(data["action128"][indices, 9])
        unique_commands.append(data["sent_commands"][indices, 6])
        case_windows = []
        if row["summary"]["pose_caps_periods"] != row["summary"]["expected_periods"]:
            pointwise_exceptions.append(index)
        for frame, image, checked_image in zip(ANCHORS, producer["images"], checker["images"], strict=True):
            receipt = producer["endpoint_receipts"][frame]
            prefix.require(
                image["current_endpoint"] == checked_image["current_endpoint"] == receipt
                and receipt == checker["endpoint_receipts"][frame]
                and receipt["endpoint_step"] == frame
                and receipt["sim_state_sha256"]
                == hashlib.sha256(np.ascontiguousarray(data["sim_states"][frame]).tobytes()).hexdigest()
                and receipt["sim_time"] == float(data["sim_time"][frame])
                and image["canvas_rgb_sha256"] == checked_image["canvas_rgb_sha256"],
                "Each real current image must bind its independently replayed endpoint",
            )
            for ref in (
                image["raw_cameras"],
                image["canvas_png"],
                checked_image["raw_cameras"],
                checked_image["canvas_png"],
            ):
                evidence.path(ref)
            canvas_check = verify_canvas(image, data, receipt)
            result = audit_window(data, frame, tool, normalization)
            observed = result.pop("observed_openings")
            commands = result.pop("actual_gripper_commands")
            flags = clipping_flags(data["action128"][frame : frame + 40], normalization)
            np.testing.assert_array_equal(flags.sum(axis=0), result["clipped_per_active_slot"])
            window_flags.append(flags)
            window_openings.append(observed)
            window_commands.append(commands)
            entry = dict(
                case_index=index,
                source_identity=identity,
                variant=spec["variant"],
                anchor_step=frame,
                pointwise_exception_case=index == 13,
                trajectory_ref=producer["arrays_ref"],
                image_ref=image["canvas_png"],
                current_endpoint=receipt,
                canvas_check=canvas_check,
                opening_range=[float(observed.min()), float(observed.max())],
                opening_below_count=int(np.count_nonzero(observed < config["opening_below"])),
                command_counts=command_counts(commands),
                **result,
            )
            windows.append(entry)
            case_windows.append(entry)
        cases.append(
            dict(
                case_index=index,
                source_identity=identity,
                variant=spec["variant"],
                pointwise_exception_case=index == 13,
                unique_transition_rows=len(indices),
                unique_clipping_per_active_slot=unique_clipping.sum(axis=0).tolist(),
                window_clipping_per_active_slot=np.sum(
                    [row["clipped_per_active_slot"] for row in case_windows], axis=0
                ).tolist(),
                unique_opening_below_count=int(
                    np.count_nonzero(data["action128"][indices, 9] < config["opening_below"])
                ),
                unique_command_counts=command_counts(data["sent_commands"][indices, 6]),
            )
        )
    prefix.require(
        len(groups) == config["source_groups"]
        and all(len(group["case_indices"]) == 4 for group in groups)
        and len(windows) == config["window_count"]
        and len(unique_rows) == config["unique_case_transition_rows"]
        and pointwise_exceptions == config["pointwise_exception_cases"],
        "All six groups, 72 windows and pointwise exception must remain fixed",
    )
    validation_identity = validate_six_sources(groups, prepared, preflight, validation, verified)
    wf, uf = np.concatenate(window_flags), np.concatenate(unique_flags)
    wo, uo = np.concatenate(window_openings), np.concatenate(unique_openings)
    wc, uc = np.concatenate(window_commands), np.concatenate(unique_commands)
    prefix.require(
        len(wf) == len(wo) == len(wc) == config["window_label_rows"]
        and len(uf) == len(uo) == len(uc) == len(unique_rows),
        "All overlapping and unique rows must be accounted for",
    )
    evidence.verify()
    feedback.require_no_simulator_import()
    return dict(
        complete=True,
        passed=True,
        config_ref=config_ref,
        input_refs=list(evidence.refs.values()),
        normalization_ref=refs["normalization"],
        validation_identity=validation_identity,
        original_validation_initial_sync_certified=prepared["exclusion"]["validation_initial_sync_certified"],
        source_groups=groups,
        cases=cases,
        windows=windows,
        registered_cases=len(cases),
        window_count=len(windows),
        window_label_rows=len(wf),
        unique_case_transition_rows=len(unique_rows),
        repeated_window_label_rows=len(wf) - len(unique_rows),
        unique_simulator_state_hash_pairs=len(state_pairs),
        state_hash_pair_includes_hidden_controller_state=False,
        window_clipping_per_active_slot=wf.sum(axis=0).tolist(),
        unique_clipping_per_active_slot=uf.sum(axis=0).tolist(),
        window_clipping_total=int(wf.sum()),
        unique_clipping_total=int(uf.sum()),
        window_opening_range=[float(wo.min()), float(wo.max())],
        unique_opening_range=[float(uo.min()), float(uo.max())],
        window_opening_below_count=int(np.count_nonzero(wo < config["opening_below"])),
        unique_opening_below_count=int(np.count_nonzero(uo < config["opening_below"])),
        window_command_counts=command_counts(wc),
        unique_command_counts=command_counts(uc),
        pointwise_exception_cases=pointwise_exceptions,
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
