"""Independently reconstruct recorded event candidates and their H40 labels."""

import hashlib
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_correct_reference as recorded
from scripts.simulation.runtime import run_reset_start_feedback as feedback
from scripts.verification.response_models.forward.verify_training_recovery_labels import check_labels
from scripts.verification.wam.training import audit_broader_recovery_events as primary
from scripts.verification.wam.training.audit_recovery_h40_contract import Evidence, load_data
from scripts.verification.wam.training.verify_broader_recovery_h40 import count_commands, scalar_normalization

ROOT = prefix.REPO
PRIMARY = ROOT / "outputs/broader_recovery_events_20261116/result.json"
OUTPUT = ROOT / "outputs/broader_recovery_events_20261116/independent.json"


def scalar_events(opening, success):
    """Reconstruct event counts and candidate anchors with scalar iteration."""
    periods = len(opening) - 1
    prefix.require(periods >= 96 and len(success) == periods + 1, "Full measured event arrays are required")
    covered = set(range(0, 40)) | set(range(16, 56)) | set(range(56, 96))
    indices = {
        "low_opening": [step for step in range(periods) if float(opening[step + 1]) < 0.35],
        "native_success": [step for step in range(periods) if bool(success[step + 1])],
    }
    summaries, candidates = {}, {}
    for kind in ("low_opening", "native_success"):
        steps = indices[kind]
        summaries[kind] = dict(
            total=len(steps),
            covered=sum(step in covered for step in steps),
            first_action_index=steps[0] if steps else None,
            last_action_index=steps[-1] if steps else None,
        )
        if steps:
            first = steps[0]
            anchor = max(0, min(periods - 40, first - 7))
            if anchor not in candidates:
                candidates[anchor] = dict(
                    anchor_step=anchor,
                    events=[],
                    current_image_available=anchor in (0, 16, 56),
                    full_h40=True,
                    selection_uses_future_training_labels=True,
                )
            candidates[anchor]["events"].append(
                dict(kind=kind, action_index=first, endpoint_index=first + 1, actual_lead_periods=first + 1 - anchor)
            )
    return dict(
        periods=periods,
        reset_success=bool(success[0]),
        covered_case_transitions=len(covered),
        low_opening=summaries["low_opening"],
        native_success=summaries["native_success"],
        candidates=[candidates[step] for step in sorted(candidates)],
    )


def compare_events(expected, case):
    prefix.require(
        all(
            case[key] == expected[key]
            for key in ("periods", "reset_success", "covered_case_transitions", "low_opening", "native_success")
        )
        and len(case["candidates"]) == len(expected["candidates"]),
        "The independently reconstructed event counts or candidate denominator differ",
    )
    for actual, candidate in zip(case["candidates"], expected["candidates"], strict=True):
        prefix.require(
            all(actual[key] == value for key, value in candidate.items()),
            "The independently selected event anchor, lead or image availability differs",
        )


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
    recovery, h40, prepared, normalization = (
        evidence.json(refs[key]) for key in ("recovery_audit", "h40_audit", "training_preparation", "normalization")
    )
    for path in (
        Path(__file__).resolve(),
        ROOT / "scripts/verification/response_models/forward/verify_training_recovery_labels.py",
        ROOT / "scripts/verification/wam/training/verify_broader_recovery_h40.py",
        ROOT / "tests/research/wam/training/test_broader_recovery_events_verify.py",
    ):
        evidence.path(prefix.reference(path))
    for ref in result["input_refs"]:
        evidence.path(ref)
    prefix.require(
        result["complete"] is result["passed"] is True
        and result["config_ref"] == config_ref
        and result["registered_cases"] == len(result["cases"]) == len(recovery["cases"]) == 24
        and h40["complete"] is h40["passed"] is True
        and result["training_authorized"] is False,
        "All 24 registered training cases and their earlier H40 audit are required",
    )
    rebuilt, flags_total, candidate_rows, union_rows = [], np.zeros(10, dtype=np.int64), set(), set()
    all_receipts = normalized_windows = 0
    for index, row in enumerate(result["cases"]):
        prior = recovery["cases"][index]
        producer, checker = (evidence.json(prior[key]) for key in ("producer_ref", "checker_ref"))
        data = load_data(evidence, producer["arrays_ref"])
        evidence.path(checker["arrays_ref"])
        identity = h40["cases"][index]["source_identity"]
        anchors = [
            item
            for item in prepared["training_anchors"]
            if item["source_identity"] == identity and item["anchor_frame"] == 0
        ]
        prefix.require(
            len(anchors) == 1
            and row["case_index"] == index
            and row["source_identity"] == identity
            and row["variant"] == prior["case"]["variant"]
            and row["trajectory_ref"] == producer["arrays_ref"]
            and prior["qualified"] is prior["replayed"] is prior["evidence_complete"] is True,
            "Each original source and replayed physical trajectory must match",
        )
        check_labels(data, np.asarray(anchors[0]["tool_source_from_tcp"], dtype=np.float64))
        periods = recorded.periods(producer)
        prefix.require(
            len(periods) == prior["case"]["periods"] == len(data["opening"]) - 1,
            "The full actual trajectory is required",
        )
        for step, period in enumerate(periods):
            prefix.require(
                period["step"] == step
                and bool(data["native_success"][step + 1]) == period["task_success"]
                and period["endpoint_receipt"]
                == producer["endpoint_receipts"][step + 1]
                == checker["endpoint_receipts"][step + 1],
                "Every actual endpoint and task predicate must match its replayed physical record",
            )
            np.testing.assert_array_equal(period["actual_state"], data["sim_states"][step + 1])
        all_receipts += len(periods)
        expected = scalar_events(data["opening"], data["native_success"])
        compare_events(expected, row)
        existing = set(range(0, 40)) | set(range(16, 56)) | set(range(56, 96))
        selected = set()
        for candidate in row["candidates"]:
            step = candidate["anchor_step"]
            receipt = producer["endpoint_receipts"][step]
            prefix.require(
                candidate["source_identity"] == identity
                and candidate["current_endpoint"] == receipt == checker["endpoint_receipts"][step]
                and receipt["sim_state_sha256"]
                == hashlib.sha256(np.ascontiguousarray(data["sim_states"][step]).tobytes()).hexdigest()
                and receipt["sim_time"] == float(data["sim_time"][step]),
                "The candidate must bind its actual independently replayed state",
            )
            if candidate["current_image_available"]:
                image = next(image for image in producer["images"] if image["frame_index"] == step)
                prefix.require(
                    candidate["existing_image_ref"] == image["canvas_png"] and image["current_endpoint"] == receipt,
                    "The reused image must match the actual endpoint",
                )
                evidence.path(image["canvas_png"])
            else:
                prefix.require(candidate["existing_image_ref"] is None, "An absent current image cannot be fabricated")
            normalized, flags = scalar_normalization(data["action128"][step : step + 40], normalization)
            opening = data["opening"][step + 1 : step + 41]
            commands = data["sent_commands"][step : step + 40, 6]
            prefix.require(
                candidate["normalized_sha256"] == hashlib.sha256(np.ascontiguousarray(normalized).tobytes()).hexdigest()
                and candidate["normalized_dtype"] == "float32"
                and candidate["normalized_shape"] == [40, 10]
                and candidate["clipped_per_active_slot"] == flags.sum(axis=0).tolist()
                and candidate["clipped_count"] == int(flags.sum())
                and candidate["opening_range"] == [float(opening.min()), float(opening.max())]
                and candidate["opening_below_count"] == int(np.count_nonzero(opening < 0.35))
                and candidate["command_counts"] == count_commands(commands),
                "Scalar normalized labels, opening and separately sent commands must match",
            )
            normalized_windows += 1
            flags_total += flags.sum(axis=0)
            selected.update(range(step, step + 40))
        candidate_rows.update((index, step) for step in selected)
        union_rows.update((index, step) for step in selected | existing)
        missing = [
            candidate["anchor_step"] for candidate in row["candidates"] if not candidate["current_image_available"]
        ]
        replay_periods = max(missing, default=0)
        prefix.require(
            row["candidate_unique_transition_rows"] == len(selected)
            and row["union_unique_transition_rows"] == len(selected | existing)
            and row["additional_unique_transition_rows"] == len(selected - existing)
            and row["proposed_replay_periods"] == replay_periods
            and row["proposed_unsaved_camera_reads"] == (replay_periods + 7) // 8,
            "The independently reconstructed image and replay budgets differ",
        )
        rebuilt.append(dict(**expected, missing=len(missing), replay_periods=replay_periods))
    total_candidates = sum(len(row["candidates"]) for row in rebuilt)
    prefix.require(
        result["total_control_periods"] == all_receipts
        and result["low_opening_total"] == sum(row["low_opening"]["total"] for row in rebuilt)
        and result["low_opening_covered"] == sum(row["low_opening"]["covered"] for row in rebuilt)
        and result["native_success_total"] == sum(row["native_success"]["total"] for row in rebuilt)
        and result["native_success_covered"] == sum(row["native_success"]["covered"] for row in rebuilt)
        and result["candidate_count"] == total_candidates == normalized_windows
        and result["candidates_needing_current_image"] == sum(row["missing"] for row in rebuilt)
        and result["candidate_window_label_rows"] == total_candidates * 40
        and result["candidate_unique_case_transition_rows"] == len(candidate_rows)
        and result["existing_and_candidate_unique_case_transition_rows"] == len(union_rows)
        and result["candidate_clipping_per_active_slot"] == flags_total.tolist()
        and result["candidate_clipping_total"] == int(flags_total.sum())
        and result["proposed_replay_control_periods"] == sum(row["replay_periods"] for row in rebuilt)
        and result["proposed_replay_inner_steps"] == 25 * sum(row["replay_periods"] for row in rebuilt)
        and result["proposed_unsaved_camera_reads"] == sum((row["replay_periods"] + 7) // 8 for row in rebuilt),
        "The independently reconstructed full event and proposed replay totals differ",
    )
    evidence.verify()
    feedback.require_no_simulator_import()
    return dict(
        complete=True,
        passed=True,
        primary_ref=result_ref,
        input_refs=list(evidence.refs.values()),
        cases_checked=len(rebuilt),
        actual_periods_checked=all_receipts,
        candidates_checked=total_candidates,
        normalized_h40_hashes_checked=normalized_windows,
        candidate_unique_case_transition_rows=len(candidate_rows),
        existing_and_candidate_unique_case_transition_rows=len(union_rows),
        independent_algorithm="scalar event search, measured endpoints and scalar FP32 H40 normalization",
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
        raise SystemExit("Independent event audit output already exists; no automatic retry or overwrite")
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
