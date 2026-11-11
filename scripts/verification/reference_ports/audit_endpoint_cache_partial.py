"""Audit completed cached replay evidence without changing its failed owner."""

import json

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_endpoint_cache_replay as runner

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/causal_stream_endpoint_cache_partial_audit_20261111/result.json"
OWNER = {
    "path": str(ROOT / "outputs/causal_stream_endpoint_cache_replay_20261111/result.json"),
    "sha256": "c14f86c7ce86035ae6b3834921f45f70b7789e29a6721fa477c361101bb5c309",
}
BRANCH = {
    "path": str(ROOT / "outputs/causal_stream_endpoint_cache_replay_20261111/simulation/result.json"),
    "sha256": "6c144e223ec50031edac3f6a54579142c9c9fb66927bed18a2423ccd21ed9a58",
}


def summarize(inputs, owner, branch):
    prefix.require(
        owner["complete"] is owner["passed"] is branch["complete"] is branch["passed"] is False
        and owner["available_branch_ref"] == BRANCH
        and branch["error"]["message"]
        == "The cached replay must complete the saved actions and one unchanged current image capture"
        and branch["new_model_requests"] == branch["new_denoise_calls"] == branch["newly_selected_commands"] == 0,
        "The original failed owner and its preserved cached replay are required",
    )
    lifecycle = prefix.read(owner["workers_ref"])
    prefix.require(
        not lifecycle["launch_error"]
        and len(lifecycle["workers"]) == 1
        and lifecycle["workers"][0]["returncode"] == 1
        and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True
        and lifecycle["workers"][0]["signal_sent"] is False,
        "The failed physical worker must have actually exited and been reaped",
    )
    runtime, capture = branch["runtime"], branch["capture"]
    replay = runtime["continuation"]
    expected_stream = inputs["base"]["stream"]
    actual = [period for row in replay["segments"] for period in row["continuation"]["periods"]]
    expected = [period for row in expected_stream["segments"] for period in row["continuation"]["periods"]]
    prefix.require(
        runtime["complete"] is runtime["passed"] is runtime["cleanup_complete"] is replay["complete"] is True
        and runtime["renderer"]["backend"] == "osmesa"
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == 120
        and runtime["newly_selected_commands"] == 0
        and runtime["limits"]["mj_step"] == 3000
        and runtime["counters"]["mj_step"]["attempted"] == runtime["counters"]["mj_step"]["completed"] == 3000
        and replay["control_steps_attempted"] == replay["control_steps_completed"] == 40
        and len(actual) == len(expected) == 40,
        "The saved runtime must contain the complete 80+40 CPU action replay",
    )
    events = branch["endpoint_cache_events"]
    observed, reused = [], []
    latest = None
    for event in events:
        if event["kind"] == "observed":
            latest = event
            observed.append(event)
        else:
            prefix.require(
                event["kind"] == "reused"
                and latest is not None
                and (event["step"], event["time"], event["state_sha256"])
                == (latest["step"], latest["time"], latest["state_sha256"]),
                "A reused endpoint must match the immediately preceding exact-state observation",
            )
            reused.append(event["step"])
    prefix.require(
        [event["step"] for event in observed] == list(range(81, 121)) and reused == [88, 96, 104, 112, 120, 120],
        "The actual endpoint observation and reuse sequence must be complete",
    )
    maxima = {"state": 0.0, "q": 0.0, "qvel": 0.0, "time": 0.0}
    for index, (left, right, event) in enumerate(zip(actual, expected, observed, strict=True)):
        np.testing.assert_array_equal(left["command"], right["command"])
        left_state, right_state = np.asarray(left["actual_state"]), np.asarray(right["actual_state"])
        endpoint = left["endpoint_receipt"]
        prefix.require(
            left["step"] == right["step"] == 80 + index
            and endpoint["endpoint_step"] == event["step"]
            and endpoint["sim_time"] == event["time"]
            and endpoint["sim_state_sha256"] == event["state_sha256"] == state_sha256(left_state)
            and len(left["actual_inner"]) == len(right["actual_inner"]) == 25
            and all(left[key] == right[key] for key in ("task_success", "target_grasp", "pose_caps_satisfied")),
            "Replayed actions, actual endpoint binding and task labels must agree",
        )
        maxima["state"] = max(maxima["state"], float(np.max(np.abs(left_state - right_state))))
        for key in ("q", "qvel", "time"):
            left_inner = np.asarray([row[key] for row in left["actual_inner"]])
            right_inner = np.asarray([row[key] for row in right["actual_inner"]])
            maxima[key] = max(maxima[key], float(np.max(np.abs(left_inner - right_inner))))
    prefix.require(
        all(value <= 1e-10 for value in maxima.values()), "Saved replay differs beyond the physical tolerance"
    )
    last = actual[-1]["endpoint_receipt"]
    captured_endpoint = capture["current_endpoint"]
    prefix.require(
        capture["frame_index"] == captured_endpoint["endpoint_step"] == 120
        and captured_endpoint["sim_state_sha256"] == last["sim_state_sha256"] == observed[-1]["state_sha256"]
        and captured_endpoint["sim_time"] == last["sim_time"] == observed[-1]["time"]
        and capture["render_before"] == capture["render_after"]
        and capture["render_before"]["sim_state_sha256"] == last["sim_state_sha256"]
        and capture["render_before"]["sim_time"] == last["sim_time"]
        and capture["future_rgb_frames"] == 0
        and capture["camera_keys"] == ["agentview_image", "robot0_eye_in_hand_image"],
        "The captured image must bind the actual unchanged replay endpoint",
    )
    prefix.checked(capture["canvas_png"])
    prefix.checked(capture["raw_cameras"])
    tail = 0
    for period in reversed(actual):
        if not period["task_success"]:
            break
        tail += 1
    prefix.require(tail == 33, "The saved task labels must preserve their observed tail")
    return {
        "complete": True,
        "passed": True,
        "original_owner_passed": False,
        "control_qualified": False,
        "replayed_actions": 40,
        "native_steps": 3000,
        "replay_max_difference": maxima,
        "task_success_tail_periods": tail,
        "observed_endpoint_calls": len(observed),
        "reused_endpoint_calls": len(reused),
        "historical_final_state_sha256": expected[-1]["endpoint_receipt"]["sim_state_sha256"],
        "replay_final_state_sha256": last["sim_state_sha256"],
        "capture_ref": capture["canvas_png"],
        "new_model_requests": 0,
        "new_physics_steps": 0,
        "owner_ref": OWNER,
        "branch_ref": BRANCH,
    }


def load_audit():
    prefix.environment()
    inputs = runner.load_inputs()
    return summarize(inputs, prefix.read(OWNER), prefix.read(BRANCH))


def main():
    print(json.dumps(publish(OUTPUT, load_audit())), flush=True)


if __name__ == "__main__":
    main()
