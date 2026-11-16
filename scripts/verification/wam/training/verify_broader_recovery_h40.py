"""Independently reconstruct measured H40 labels and FP32 clipping results."""

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.research.wam.training.clean.clean_training_data import VALIDATION_DEMO_KEYS
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_correct_reference as recorded
from scripts.simulation.runtime import run_reset_start_feedback as feedback
from scripts.verification.response_models.forward.verify_training_recovery_labels import check_labels
from scripts.verification.wam.training import audit_broader_recovery_h40 as primary
from scripts.verification.wam.training.audit_recovery_h40_contract import Evidence, load_data

ROOT = prefix.REPO
PRIMARY = ROOT / "outputs/broader_recovery_h40_20261116/result.json"
OUTPUT = ROOT / "outputs/broader_recovery_h40_20261116/independent.json"


def scalar_normalization(action128, normalization):
    """Use explicit scalar FP32 multiply/add/clip in the frozen mask order."""
    actions = np.asarray(action128, dtype=np.float64)
    prefix.require(
        actions.ndim == 2 and actions.shape[1] == 128 and np.isfinite(actions).all(), "Finite action rows required"
    )
    transform = normalization["transform"]
    mask = np.asarray(transform["normalization_mask"])
    prefix.require(
        mask.shape == (128,) and mask.dtype == np.bool_ and not mask[10:].any(), "Frozen active mask required"
    )
    low, high = np.float32(transform["clip_min"]), np.float32(transform["clip_max"])
    output = np.empty((len(actions), 10), dtype=np.float32)
    flags = np.zeros((len(actions), 10), dtype=bool)
    for row in range(len(actions)):
        for slot in range(10):
            value = np.float32(actions[row, slot])
            if mask[slot]:
                product = np.float32(value * np.float32(transform["scale"][slot]))
                value = np.float32(product + np.float32(transform["bias"][slot]))
                flags[row, slot] = value < low or value > high
                value = np.clip(value, low, high)
            output[row, slot] = value
    prefix.require(np.isfinite(output).all(), "Finite scalar normalized values are required")
    return output, flags


def check_source_separation(groups, training, preflight, validation, verification):
    """Independently compare training, validation and fresh demo/SHA identities."""
    fresh = training["exclusion"]["fresh_sources"]
    records, prepared = preflight["records"], validation["records"]

    def identities(rows):
        return {(item["task_index"], item["demo"], item["initial_state_sha256"]) for item in rows}

    def keys(rows):
        return {(task, demo) for task, demo, _ in rows}

    def hashes(rows):
        return {sha for _, _, sha in rows}

    train_ids, val_ids, fresh_ids = identities(groups), identities(records), identities(fresh)
    prepared_ids = identities(prepared)
    expected_names = {f"task{task}_validation_{demo}" for task, demo in keys(val_ids)}
    forbidden = {tuple(item) for item in training["exclusion"]["forbidden_demo_keys"]}
    forbidden_hashes = set(training["exclusion"]["forbidden_initial_state_sha256"])
    training_anchors = {
        (
            row["source_identity"]["task_index"],
            row["source_identity"]["demo"],
            row["source_identity"]["initial_state_sha256"],
        )
        for row in training["training_anchors"]
        if row["anchor_frame"] == 0
    }
    sets = (train_ids, val_ids, fresh_ids)
    prefix.require(
        len(groups) == len(train_ids) == len(keys(train_ids)) == len(hashes(train_ids)) == 6
        and all(sum(row["task_index"] == task for row in groups) == 2 for task in range(3))
        and train_ids <= training_anchors
        and len(records) == len(prepared) == len(val_ids) == len(keys(val_ids)) == len(hashes(val_ids)) == 6
        and val_ids == prepared_ids
        and keys(val_ids) == VALIDATION_DEMO_KEYS
        and len(fresh) == len(fresh_ids) == len(keys(fresh_ids)) == len(hashes(fresh_ids)) == 12
        and all(
            not (keys(sets[i]) & keys(sets[j])) and not (hashes(sets[i]) & hashes(sets[j]))
            for i in range(3)
            for j in range(i + 1, 3)
        )
        and not keys(train_ids) & forbidden
        and not hashes(train_ids) & forbidden_hashes
        and preflight["complete"] is preflight["passed"] is preflight["training_fresh_identity_disjoint"] is True
        and verification["complete"] is verification["passed"] is verification["eligible_for_model_validation"] is True
        and verification["fresh_authorized"] is False
        and verification["native_stable_cases"] == 6
        and verification["physics_endpoints_verified"] == 1218
        and verification["current_images_verified"] == 24
        and len(verification["cases"]) == 6
        and {item["name"] for item in verification["cases"]} == expected_names
        and all(item["native_stable_last40"] is True for item in verification["cases"]),
        "Independent training, validation and fresh identities must be disjoint",
    )


def count_commands(values):
    return dict(sorted(Counter(str(0.0 if value == 0 else float(value)) for value in values).items()))


def verify():
    prefix.environment()
    feedback.require_no_simulator_import()
    evidence = Evidence()
    result_ref = prefix.reference(PRIMARY)
    result = evidence.json(result_ref)
    config_ref = prefix.reference(primary.CONFIG)
    config = evidence.json(config_ref)
    primary.validate_config(config)
    refs = {key: preparation.configured_ref(config[key]) for key in primary.REFERENCES}
    recovery, training, normalization, preflight, validation, verification = (
        evidence.json(refs[key])
        for key in (
            "recovery_audit",
            "training_preparation",
            "normalization",
            "validation_preflight",
            "validation_prepared",
            "validation_verified",
        )
    )
    for path in (
        Path(__file__).resolve(),
        ROOT / "scripts/verification/response_models/forward/verify_training_recovery_labels.py",
        ROOT / "tests/research/wam/training/test_broader_recovery_h40_verify.py",
    ):
        evidence.path(prefix.reference(path))
    for ref in result["input_refs"]:
        evidence.path(ref)
    prefix.require(
        result["complete"] is result["passed"] is True
        and result["config_ref"] == config_ref
        and result["training_authorized"] is False
        and result["registered_cases"] == len(result["cases"]) == 24
        and result["window_count"] == len(result["windows"]) == 72
        and result["pointwise_exception_cases"] == [13]
        and preflight["prepared_ref"] == refs["training_preparation"]
        and validation["preflight_ref"] == refs["validation_preflight"]
        and verification["prepared_ref"] == refs["validation_prepared"],
        "The registered 24 cases, validation chain and 72 windows are required",
    )
    check_source_separation(result["source_groups"], training, preflight, validation, verification)
    unique_pairs, window_flags, unique_flags, window_openings, unique_openings = set(), [], [], [], []
    window_commands, unique_commands, violations = [], [], []
    for index, case in enumerate(recovery["cases"]):
        spec = case["case"]
        producer, checker = (evidence.json(case[key]) for key in ("producer_ref", "checker_ref"))
        data = load_data(evidence, producer["arrays_ref"])
        evidence.path(checker["arrays_ref"])
        identity = result["cases"][index]["source_identity"]
        original = [
            row
            for row in training["training_anchors"]
            if row["source_identity"] == identity and row["anchor_frame"] == 0
        ]
        prefix.require(
            len(original) == 1
            and case["case"]["index"] == result["cases"][index]["case_index"] == index
            and producer["case"] == checker["case"] == spec
            and case["qualified"] is case["replayed"] is case["evidence_complete"] is True,
            "Each independently replayed case and training identity must match",
        )
        tool = np.asarray(original[0]["tool_source_from_tcp"], dtype=np.float64)
        check_labels(data, tool)
        np.testing.assert_allclose(
            data["source_pose"][:, :3, 3], data["actual_source_site_pose"][:, :3, 3], atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            data["source_pose"][:, :3, :3], data["actual_eef_body_pose"][:, :3, :3], atol=1e-10, rtol=0
        )
        for period in recorded.periods(producer):
            if not period["pose_caps_satisfied"]:
                violations.append((index, period["step"]))
        indices = sorted(set(range(0, 40)) | set(range(16, 56)) | set(range(56, 96)))
        hashes = [hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest() for state in data["sim_states"]]
        unique_pairs.update((hashes[step], hashes[step + 1]) for step in indices)
        _, unique_case_flags = scalar_normalization(data["action128"][indices], normalization)
        unique_flags.append(unique_case_flags)
        unique_openings.extend(data["action128"][indices, 9].tolist())
        unique_commands.extend(data["sent_commands"][indices, 6].tolist())
        claimed_case = result["cases"][index]
        prefix.require(
            claimed_case["unique_transition_rows"] == 96
            and claimed_case["unique_clipping_per_active_slot"] == unique_case_flags.sum(axis=0).tolist()
            and claimed_case["unique_opening_below_count"]
            == int(np.count_nonzero(data["action128"][indices, 9] < 0.35))
            and claimed_case["unique_command_counts"] == count_commands(data["sent_commands"][indices, 6]),
            "Per-case unique transition statistics differ",
        )
        for position, step in enumerate(primary.ANCHORS):
            claimed = result["windows"][index * 3 + position]
            image, replay = producer["images"][position], checker["images"][position]
            receipt = producer["endpoint_receipts"][step]
            prefix.require(
                claimed["case_index"] == index
                and claimed["anchor_step"] == step
                and claimed["source_identity"] == identity
                and claimed["trajectory_ref"] == producer["arrays_ref"]
                and claimed["image_ref"] == image["canvas_png"]
                and claimed["current_endpoint"] == receipt == image["current_endpoint"] == replay["current_endpoint"]
                and receipt["sim_state_sha256"] == hashes[step]
                and receipt["sim_time"] == float(data["sim_time"][step])
                and image["canvas_rgb_sha256"] == replay["canvas_rgb_sha256"],
                "The actual and independently replayed current image endpoint must match",
            )
            for ref in (image["canvas_png"], image["raw_cameras"], replay["canvas_png"], replay["raw_cameras"]):
                evidence.path(ref)
            normalized, flags = scalar_normalization(data["action128"][step : step + 40], normalization)
            digest = hashlib.sha256(np.ascontiguousarray(normalized).tobytes()).hexdigest()
            observed = data["opening"][step + 1 : step + 41]
            sent = data["sent_commands"][step : step + 40, 6]
            prefix.require(
                claimed["normalized_sha256"] == digest
                and claimed["normalized_dtype"] == "float32"
                and claimed["normalized_shape"] == [40, 10]
                and claimed["clipped_per_active_slot"] == flags.sum(axis=0).tolist()
                and claimed["clipped_count"] == int(flags.sum())
                and claimed["opening_range"] == [float(observed.min()), float(observed.max())]
                and claimed["opening_below_count"] == int(np.count_nonzero(observed < 0.35))
                and claimed["command_counts"] == count_commands(sent),
                "Scalar FP32 normalized labels, measured opening or actual commands differ",
            )
            window_flags.append(flags)
            window_openings.extend(observed.tolist())
            window_commands.extend(sent.tolist())
    wf, uf = np.concatenate(window_flags), np.concatenate(unique_flags)
    prefix.require(
        violations == [(13, 295)]
        and len(wf) == len(window_openings) == len(window_commands) == 2880
        and len(uf) == len(unique_openings) == len(unique_commands) == 2304
        and result["window_label_rows"] == len(wf)
        and result["unique_case_transition_rows"] == len(uf)
        and result["repeated_window_label_rows"] == len(wf) - len(uf)
        and result["unique_simulator_state_hash_pairs"] == len(unique_pairs)
        and result["window_clipping_per_active_slot"] == wf.sum(axis=0).tolist()
        and result["unique_clipping_per_active_slot"] == uf.sum(axis=0).tolist()
        and result["window_clipping_total"] == int(wf.sum())
        and result["unique_clipping_total"] == int(uf.sum())
        and result["window_opening_range"] == [min(window_openings), max(window_openings)]
        and result["unique_opening_range"] == [min(unique_openings), max(unique_openings)]
        and result["window_opening_below_count"] == sum(value < 0.35 for value in window_openings)
        and result["unique_opening_below_count"] == sum(value < 0.35 for value in unique_openings)
        and result["window_command_counts"] == count_commands(window_commands)
        and result["unique_command_counts"] == count_commands(unique_commands),
        "The complete independent label, clipping, source and exception totals must agree",
    )
    evidence.verify()
    feedback.require_no_simulator_import()
    return dict(
        complete=True,
        passed=True,
        primary_ref=result_ref,
        input_refs=list(evidence.refs.values()),
        cases_checked=24,
        windows_checked=72,
        normalized_hashes_checked=72,
        window_label_rows_checked=len(wf),
        unique_case_transition_rows_checked=len(uf),
        pointwise_pose_violations=[dict(case_index=index, step=step) for index, step in violations],
        independent_algorithm="direct full-trajectory TCP deltas; scalar FP32 multiply/add/clip; source metadata sets",
        unknown_near_duplicates_excluded=False,
        validation_future_labels_read=False,
        fresh_future_labels_read=False,
        training_authorized=False,
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
        fits=0,
        fresh_cases=0,
    )


def main():
    if OUTPUT.exists():
        raise SystemExit("Independent audit output already exists; no automatic retry or overwrite")
    try:
        result = verify()
    except Exception as error:
        publish_result(
            OUTPUT, dict(complete=False, passed=False, error=dict(type=type(error).__name__, message=str(error)))
        )
        raise
    print(publish_result(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()
