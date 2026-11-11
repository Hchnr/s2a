"""Audit early shared-reference branches and classify saved contact pairs."""

from collections import Counter
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import prepare_early_box_face_feedback as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime.run_early_box_face_feedback import compare_common_reference
from scripts.verification.response_models.audit_box_face_feedback import assert_same_observation, metrics

ROOT = prefix.REPO
OWNER = ROOT / "outputs/early_box_face_feedback_20261111/result.json"
OWNER_SHA = "57cb3e4d35ecb5ba7f6e13d84006c11b4ad4b35cc7bb359ab71ec6071af69a20"
OUTPUT = ROOT / "outputs/early_box_face_feedback_audit_20261111/result.json"


def contact_pair_counts(samples):
    """Count each active robot/environment pair at most once per inner sample."""
    counts = Counter()
    for sample in samples:
        pairs = {
            tuple(sorted((contact["geom1"], contact["geom2"])))
            for contact in sample["contacts"]
            if contact["active_constraint"] and contact["kind"] == "robot_environment"
        }
        counts.update(pairs)
    return [{"geometries": list(pair), "active_inner_samples": count} for pair, count in sorted(counts.items())]


def audit_replay(producer, checker):
    prefix.require(
        producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True
        and checker["candidate_modules_unloaded"] is True,
        "A complete producer and policy-isolated checker are required",
    )
    first, second = producer["runtime"]["continuation"], checker["runtime"]["continuation"]
    compare_common_reference(first, second)
    left, right = first["probe"]["periods"], second["probe"]["periods"]
    prefix.require(
        len(left) == len(right) and len(first["inner_samples"]) == len(second["inner_samples"]) == 25 * len(left),
        "Each completed period requires all replayed inner samples",
    )
    maximum = 0.0
    for produced, replayed in zip(left, right, strict=True):
        np.testing.assert_array_equal(produced["command"], replayed["command"])
        difference = float(np.max(np.abs(np.asarray(produced["actual_state"]) - replayed["actual_state"])))
        prefix.require(difference <= 1e-10, "The replayed state must match")
        maximum = max(maximum, difference)
        assert_same_observation(produced["actual_inner"], replayed["actual_inner"], path="period.inner")
        assert_same_observation(produced["completed_feedback"], replayed["completed_feedback"], path="period.feedback")
        for key in (
            "pose_caps_satisfied",
            "task_success",
            "target_grasp",
            "inner_outside_samples",
            "inner_below_buffer_samples",
        ):
            prefix.require(produced[key] == replayed[key], f"Replay label differs: {key}")
    assert_same_observation(first["inner_samples"], second["inner_samples"], path="inner_samples")
    return {"periods": len(left), "inner_samples": len(first["inner_samples"]), "max_state_difference": maximum}


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
            and len(lifecycle["workers"]) == 2
            and all(
                row["actual_wait"] is row["reaped"] is True and row["returncode"] == 0 and row["signal_sent"] is False
                for row in lifecycle["workers"]
            ),
            "Every owned worker must exit and be reaped",
        )
    modes = ("original", "box_face")
    producers = {mode: prefix.read(owner["producer_refs"][mode]) for mode in modes}
    checkers = {mode: prefix.read(owner["checker_refs"][mode]) for mode in modes}
    replay = {mode: audit_replay(producers[mode], checkers[mode]) for mode in modes}
    compare_common_reference(*(producers[mode]["runtime"]["continuation"] for mode in modes))
    config = inputs["early_box_face_feedback"]["config"]
    original = producers["original"]["runtime"]["continuation"]
    prefix.require(
        original["branch_start"]["state_sha256"] == config["branch_state_sha256"]
        and original["reference_image_ref"]["sha256"] == config["reference_image_sha256"]
        and original["reference_response_ref"]["sha256"] == config["reference_response_sha256"],
        "The shared start and prior reference must match the registered source",
    )
    n = min(replay[mode]["periods"] for mode in modes)
    prefix.require(n == owner["matched_periods"] and n <= 8, "The matched period count must use the common prefix")
    matched, pairs = {}, {}
    for mode in modes:
        continuation = producers[mode]["runtime"]["continuation"]
        matched[mode] = metrics(continuation, n)
        for field, value in matched[mode].items():
            assert_same_observation(value, owner["matched_summaries"][mode][field], path=f"{mode}.{field}")
        pairs[mode] = contact_pair_counts(continuation["inner_samples"][: 25 * n])
    native_steps = sum(
        branch["runtime"]["counters"]["mj_step"]["completed"]
        for group in (producers, checkers)
        for branch in group.values()
    )
    commands = sum(producers[mode]["runtime"]["newly_selected_commands"] for mode in modes)
    prefix.require(
        native_steps == owner["new_native_steps"] <= 12600
        and commands == owner["new_selected_commands"] <= 16
        and owner["new_world_model_calls"] == owner["fresh_cases"] == 0,
        "The registered physical and model budgets must hold",
    )
    sources = [
        prefix.reference(Path(__file__).resolve()),
        prefix.reference(ROOT / "tests/research/native_runtime/test_early_box_face_feedback_audit.py"),
    ]
    print(
        publish(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                control_qualified=False,
                owner_ref=owner_ref,
                source_refs=sources,
                common_start_and_reference=True,
                replay=replay,
                matched=matched,
                active_robot_environment_pairs=pairs,
                new_native_steps=native_steps,
                new_selected_commands=commands,
                new_world_model_calls=0,
                fresh_cases=0,
                policy_or_stop_independently_recomputed=False,
                scope="saved receipts and replay only",
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    execute()
