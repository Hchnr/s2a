"""Audit the saved partial cached replay without rerunning a simulator."""

import argparse
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import adaptive_long_cached_checker_admission as registration
from scripts.simulation.runtime import audit_adaptive_long_partial as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as capture_check

FAILURE = {
    "path": str(registration.OUTPUT / "checker/result.json"),
    "sha256": "83375f747c6e910d5f8d9d9c3c82b48652f8dcceaa7b825876fabd97002ab171",
}
BRANCH = {
    "path": str(registration.OUTPUT / "checker/simulation/result.json"),
    "sha256": "3430e619bc5ce1587c0aff8f1d34917ab4af2835c52841b40bd9b62bcf9d69af",
}
WORKERS = {
    "path": str(registration.OUTPUT / "checker/workers.json"),
    "sha256": "10a5ae60337cd57aab55e0262fed2f4ba3752b9f13d59e91a0449340c104fb35",
}
ERROR = "The independent current image or endpoint differs"


def compare_partial_periods(produced, replayed):
    """Report numerical differences without treating changed state hashes as equal."""
    prefix.require(
        len(produced) == len(replayed) == 12
        and [row["step"] for row in produced] == [row["step"] for row in replayed] == list(range(107, 119)),
        "Exactly the twelve completed adaptive actions are required",
    )
    maxima = {name: 0.0 for name in ("command", "state", "tcp_error", "inner_q", "inner_qvel", "inner_time", "gripper")}
    mismatch = []

    def compare(name, first, second):
        left, right = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
        prefix.require(
            left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all(),
            "The saved samples must have matching finite shapes",
        )
        delta = float(np.max(np.abs(left - right))) if left.size else 0.0
        maxima[name] = max(maxima[name], delta)

    for left, right in zip(produced, replayed, strict=True):
        prefix.require(
            len(left["actual_inner"]) == len(right["actual_inner"]) == 25
            and all(
                left[key] == right[key]
                for key in (
                    "pose_caps_satisfied",
                    "target_grasp",
                    "task_success",
                    "inner_outside_samples",
                    "inner_below_buffer_samples",
                )
            ),
            "Each completed action must preserve inner counts and recorded labels",
        )
        a, b = left["endpoint_receipt"], right["endpoint_receipt"]
        prefix.require(
            set(a) == set(b)
            and all(a[key] == b[key] for key in a if key != "sim_state_sha256")
            and a["endpoint_step"] == left["step"] + 1
            and state_sha256(np.asarray(left["actual_state"], dtype=np.float64)) == a["sim_state_sha256"]
            and state_sha256(np.asarray(right["actual_state"], dtype=np.float64)) == b["sim_state_sha256"],
            "Endpoint metadata other than the state digest must agree",
        )
        if a["sim_state_sha256"] != b["sim_state_sha256"]:
            mismatch.append(a["endpoint_step"])
        for name, key in (("command", "command"), ("state", "actual_state"), ("tcp_error", "tcp_error")):
            compare(name, left[key], right[key])
        for name, key in (("inner_q", "q"), ("inner_qvel", "qvel"), ("inner_time", "time")):
            compare(name, [row[key] for row in left["actual_inner"]], [row[key] for row in right["actual_inner"]])
        for key in ("opening", "opening_reference", "opening_error", "qpos", "qvel", "motor_target"):
            compare("gripper", left["completed_feedback"]["score"][key], right["completed_feedback"]["score"][key])
    prefix.require(
        mismatch == list(range(112, 120)) and maxima["command"] == 0 and max(maxima.values()) < 1e-10,
        "The observed exact-hash divergence and numerical differences must remain unchanged",
    )
    return dict(
        completed_actions=12,
        completed_inner_samples=300,
        differing_state_digest_endpoints=mismatch,
        max_absolute_difference=maxima,
        task_and_grasp_labels_match=True,
        strict_endpoint_identity_matches=False,
    )


def audit():
    """Bind failure, partial costs, cache events and actual image bytes."""
    prefix.environment()
    failed, branch, workers = (prefix.read(ref) for ref in (FAILURE, BRANCH, WORKERS))
    producer = prefix.read(prefix.read(previous.PRODUCER)["branch_ref"])
    prior = prefix.read(registration.PARTIAL_AUDIT)
    runtime, replay = branch["runtime"], branch["runtime"]["adaptive_replay_partial"]
    prefix.require(
        failed["complete"] is failed["passed"] is branch["complete"] is branch["passed"] is False
        and failed["automatic_retry"] is False
        and failed["partial_source_match"] is True
        and failed["available_branch_ref"] == BRANCH
        and failed["workers_ref"] == WORKERS
        and branch["error"]["message"] == runtime["error"]["message"] == replay["error"]["message"] == ERROR
        and runtime["cleanup_complete"] is True
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == 119
        and runtime["counters"]["mj_step"] == dict(attempted=2975, completed=2975)
        and runtime["counters"]["model_construct"] == runtime["counters"]["context"] == dict(attempted=3, completed=3)
        and replay["control_steps_attempted"]
        == replay["control_steps_completed"]
        == replay["replay_periods_committed"]
        == 12
        and replay["complete"] is False
        and prior["complete"] is prior["passed"] is True
        and prior["details"]["checker_complete"] is False,
        "The recorded failure and completed physical work must remain exact",
    )
    row = workers["workers"]
    prefix.require(
        workers["launch_error"] is None
        and len(row) == 1
        and row[0]["key"] == "checker"
        and row[0]["returncode"] == 1
        and row[0]["actual_wait"] is row[0]["reaped"] is True
        and row[0]["signal_sent"] is False,
        "The failed worker must be actually reaped",
    )
    prefix.checked(row[0]["log_ref"])
    prefix.require(
        failed["phase_physical_bounds"]
        == {
            key: dict(lower=count, upper=count, evidence="final_receipt")
            for key, count in dict(control_steps=119, native_steps=2975, models=3, contexts=3).items()
        }
        and failed["cumulative_physical_bounds"]
        == dict(
            control_steps=dict(lower=2174, upper=2289),
            native_steps=dict(lower=54350, upper=57225),
            models=dict(lower=93, upper=96),
            contexts=dict(lower=93, upper=96),
        )
        and failed["cumulative_model_bounds"] == prior["cumulative_model_bounds"]
        and failed["model_initializations"] == failed["model_requests"] == failed["denoise_calls"] == 0,
        "All new and historical costs must remain accounted for",
    )
    produced = [
        period
        for segment in producer["runtime"]["continuation"]["adaptive"]["segments"]
        for period in segment["continuation"]["periods"]
    ]
    replayed = [period for segment in replay["segments"] for period in segment["continuation"]["periods"]]
    differences = compare_partial_periods(produced[:12], replayed)
    events = runtime["adaptive_cached_endpoint_events"]
    prefix.require(
        [(event["step"], event["kind"]) for event in events if event["kind"] == "reused"]
        == [(111, "reused"), (119, "reused")],
        "Both recorded repeated endpoint reads must be preserved",
    )
    expected = producer["runtime"]["adaptive_captures"]
    actual = runtime["adaptive_checker_captures"]
    prefix.require(
        len(actual) == 2 and [row["frame_index"] for row in actual] == [107, 111], "Two completed captures required"
    )
    for left, right in zip(expected[:2], actual, strict=True):
        capture_check.compare_capture(left, right)
        for name in ("raw_cameras", "canvas_png"):
            prefix.checked(left[name])
            prefix.checked(right[name])
            prefix.require(left[name]["sha256"] == right[name]["sha256"], "Completed image bytes differ")
    third = expected[2]
    root = registration.OUTPUT / "checker/simulation/runtime/anchor0119"
    images = {
        "raw_cameras": prefix.reference(root / "current_cameras.npz"),
        "canvas_png": prefix.reference(root / "current.png"),
    }
    for name, actual_ref in images.items():
        prefix.checked(third[name])
        prefix.require(actual_ref["sha256"] == third[name]["sha256"], "The failed capture's saved image bytes differ")
    prefix.require(
        third["frame_index"] == events[-1]["step"] == 119
        and third["current_endpoint"]["sim_state_sha256"] != events[-1]["state_sha256"],
        "The saved endpoint digest difference at the failed capture is required",
    )
    return dict(
        complete=True,
        passed=True,
        evidence_level="saved_partial_cached_checker_audit",
        control_qualified=False,
        failed_checker_ref=FAILURE,
        failed_branch_ref=BRANCH,
        workers_ref=WORKERS,
        audit_source_ref=prefix.reference(Path(__file__).resolve()),
        partial_comparison=differences,
        reused_endpoint_steps=[111, 119],
        completed_capture_steps=[107, 111],
        failed_capture_step=119,
        failed_capture_image_refs=images,
        failed_capture_image_bytes_match=True,
        failed_capture_state_digest_matches=False,
        failure_cause_proven=False,
        cumulative_physical_bounds=failed["cumulative_physical_bounds"],
        cumulative_model_bounds=failed["cumulative_model_bounds"],
        new_model_requests=0,
        new_physics_steps=0,
        automatic_retry=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = (prefix.REPO / args.output).resolve()
    prefix.require(output.is_relative_to(prefix.REPO / "outputs"), "The audit output must stay in this repository")
    print(publish(output, audit()), flush=True)


if __name__ == "__main__":
    main()
