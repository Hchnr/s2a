"""Audit same-reference physical feedback and replay from saved receipts only."""

from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import prepare_box_face_feedback as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
OWNER = ROOT / "outputs/box_face_feedback_20261111/result.json"
OWNER_SHA = "d29173ca73201769e4ecb9765da7023429e377c1dd8ef7bd7ab9530c883f5088"
OUTPUT = ROOT / "outputs/box_face_feedback_audit_20261111/result.json"


def assert_same_observation(left, right, *, path="sample"):
    """Compare all saved sample fields with fixed numeric tolerance."""
    if isinstance(left, dict) and isinstance(right, dict):
        prefix.require(set(left) == set(right), f"Fields differ at {path}")
        for key in left:
            assert_same_observation(left[key], right[key], path=f"{path}.{key}")
    elif isinstance(left, list) and isinstance(right, list):
        prefix.require(len(left) == len(right), f"Lengths differ at {path}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_same_observation(a, b, path=f"{path}[{index}]")
    elif type(left) in (int, float) and type(right) in (int, float):
        np.testing.assert_allclose(left, right, atol=1e-9, rtol=0, err_msg=path)
    else:
        prefix.require(left == right, f"Values differ at {path}")


def same_reference(original, candidate):
    """Require the same state, image, model response and decoded targets."""
    for key in ("model_state_sha256", "model_image_ref", "model_response_ref"):
        prefix.require(original[key] == candidate[key], f"The {key} must match the baseline")
    np.testing.assert_array_equal(original["original_targets"], candidate["original_targets"])


def count_active(samples, pair=None):
    count = 0
    for sample in samples:
        matched = any(
            contact["active_constraint"]
            and (
                contact["kind"] == "robot_environment" if pair is None else {contact["geom1"], contact["geom2"]} == pair
            )
            for contact in sample["contacts"]
        )
        count += int(matched)
    return count


def audit_replay(producer, checker):
    prefix.require(
        producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True
        and checker["candidate_modules_unloaded"] is True,
        "A completed producer and policy-isolated checker are required",
    )
    first, second = producer["runtime"]["continuation"], checker["runtime"]["continuation"]
    same_reference(first, second)
    left, right = first["probe"]["periods"], second["probe"]["periods"]
    prefix.require(
        len(left) == len(right) and len(first["inner_samples"]) == len(second["inner_samples"]) == 25 * len(left),
        "The same complete periods and samples are required",
    )
    maximum_state_difference = 0.0
    for produced, checked in zip(left, right, strict=True):
        np.testing.assert_array_equal(produced["command"], checked["command"])
        delta = float(np.max(np.abs(np.asarray(produced["actual_state"]) - checked["actual_state"])))
        prefix.require(delta <= 1e-10, "Replay state differs")
        maximum_state_difference = max(maximum_state_difference, delta)
        assert_same_observation(produced["actual_inner"], checked["actual_inner"], path="period.inner")
        assert_same_observation(produced["completed_feedback"], checked["completed_feedback"], path="period.feedback")
        for key in (
            "pose_caps_satisfied",
            "task_success",
            "target_grasp",
            "inner_outside_samples",
            "inner_below_buffer_samples",
        ):
            prefix.require(produced[key] == checked[key], f"Replay label differs: {key}")
    assert_same_observation(first["inner_samples"], second["inner_samples"], path="inner_samples")
    return {
        "periods": len(left),
        "inner_samples": len(first["inner_samples"]),
        "max_state_difference": maximum_state_difference,
    }


def metrics(continuation, n):
    periods, samples = continuation["probe"]["periods"][:n], continuation["inner_samples"][: 25 * n]
    prefix.require(len(periods) == n and len(samples) == 25 * n, "All matched comparison samples required")
    positions = [float(np.linalg.norm(np.asarray(row["tcp_error"])[:3])) for row in periods]
    return {
        "periods": n,
        "original_pose_caps_satisfied": sum(row["pose_caps_satisfied"] for row in periods),
        "selected_pose_caps_satisfied": sum(
            row["completed_feedback"]["projected_pose_caps_satisfied"] for row in periods
        ),
        "mean_original_position_error_m": float(np.mean(positions)) if positions else None,
        "max_original_position_error_m": max(positions, default=None),
        "table_finger_active_contact_inner_samples": count_active(
            samples, {"table_collision", "gripper0_finger2_collision"}
        ),
        "stove_finger_active_contact_inner_samples": count_active(
            samples, {"flat_stove_1_collision_burner", "gripper0_finger2_collision"}
        ),
        "robot_environment_active_contact_inner_samples": count_active(samples),
        "task_success_endpoints": sum(row["task_success"] for row in periods),
        "target_grasp_endpoints": sum(row["target_grasp"] for row in periods),
    }


def execute():
    inputs = admission.load_inputs()
    owner_ref = {"path": str(OWNER), "sha256": OWNER_SHA}
    owner = prefix.read(owner_ref)
    prefix.require(
        owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True
        and owner["control_qualified"] is False,
        "A closed physical owner is required",
    )
    for group in owner["groups"]:
        lifecycle = prefix.read(group["workers_ref"])
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 1
            and all(
                row["actual_wait"] is row["reaped"] is True and row["returncode"] == 0 and row["signal_sent"] is False
                for row in lifecycle["workers"]
            ),
            "Both owned workers must exit and be reaped",
        )
    producer, checker = (prefix.read(owner[key]) for key in ("producer_ref", "checker_ref"))
    baseline = inputs["box_face_feedback"]["baseline_producer"]
    replay = audit_replay(producer, checker)
    candidate = producer["runtime"]["continuation"]
    original = baseline["runtime"]["continuation"]
    same_reference(original, candidate)
    n = replay["periods"]
    prefix.require(
        n == owner["matched_periods"] and 0 <= n <= 6 and owner["new_selected_commands"] == n,
        "The matched physical period budget is required",
    )
    results = {"original": metrics(original, n), "box_face": metrics(candidate, n)}
    for mode, key in (("original", "baseline_summary"), ("box_face", "producer_summary")):
        for field, value in results[mode].items():
            assert_same_observation(value, owner[key][field], path=f"{mode}.{field}")
    native_steps = sum(row["runtime"]["counters"]["mj_step"]["completed"] for row in (producer, checker))
    prefix.require(
        native_steps == owner["new_native_steps"]
        and native_steps <= 6300
        and owner["new_world_model_calls"] == owner["fresh_cases"] == 0,
        "The physical budget and no-new-model condition must hold",
    )
    source_refs = [
        prefix.reference(Path(__file__).resolve()),
        prefix.reference(ROOT / "tests/research/native_runtime/test_box_face_feedback_audit.py"),
    ]
    print(
        publish(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                control_qualified=False,
                owner_ref=owner_ref,
                source_refs=source_refs,
                same_state_image_response_and_targets=True,
                replay=replay,
                matched=results,
                new_native_steps=native_steps,
                new_world_model_calls=0,
                fresh_cases=0,
                policy_or_stop_independently_recomputed=False,
                scope="saved receipts and replay only; no new physical integration or policy recomputation",
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    execute()
