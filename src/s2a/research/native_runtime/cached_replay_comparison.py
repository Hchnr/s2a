"""Compare cached native replay to saved actions and its own captured endpoint."""

import numpy as np

from s2a.research.wam.training.native.native_training import state_sha256


def compare_cached_replay(runtime, expected_stream, events, capture):
    """Check exact in-run binding and the registered 1e-10 cross-run tolerance."""
    replay = runtime["continuation"]
    actual = [period for row in replay["segments"] for period in row["continuation"]["periods"]]
    expected = [period for row in expected_stream["segments"] for period in row["continuation"]["periods"]]
    observed = [event for event in events if event["kind"] == "observed"]
    reused = [event["step"] for event in events if event["kind"] == "reused"]
    latest = None
    for event in events:
        if event["kind"] == "observed":
            latest = event
        elif not (
            event["kind"] == "reused"
            and latest is not None
            and (event["step"], event["time"], event["state_sha256"])
            == (latest["step"], latest["time"], latest["state_sha256"])
        ):
            raise ValueError("Each reused endpoint must match the latest exact-state observation")
    if not (
        runtime["complete"] is runtime["passed"] is runtime["cleanup_complete"] is replay["complete"] is True
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == 120
        and runtime["newly_selected_commands"] == 0
        and runtime["limits"]["mj_step"] == 3000
        and runtime["counters"]["mj_step"]["attempted"] == runtime["counters"]["mj_step"]["completed"] == 3000
        and replay["control_steps_attempted"] == replay["control_steps_completed"] == 40
        and len(actual) == len(expected) == len(observed) == 40
        and [event["step"] for event in observed] == list(range(81, 121))
        and reused == [88, 96, 104, 112, 120, 120]
    ):
        raise ValueError("The cached replay must complete all saved actions and endpoint observations")
    maxima = {"state": 0.0, "q": 0.0, "qvel": 0.0, "time": 0.0}
    for index, (left, right, event) in enumerate(zip(actual, expected, observed, strict=True)):
        np.testing.assert_array_equal(left["command"], right["command"])
        left_state, right_state = np.asarray(left["actual_state"]), np.asarray(right["actual_state"])
        endpoint = left["endpoint_receipt"]
        if not (
            left["step"] == right["step"] == 80 + index
            and endpoint["endpoint_step"] == event["step"]
            and endpoint["sim_time"] == event["time"]
            and endpoint["sim_state_sha256"] == event["state_sha256"] == state_sha256(left_state)
            and len(left["actual_inner"]) == len(right["actual_inner"]) == 25
            and np.isfinite(right_state).all()
            and all(left[key] == right[key] for key in ("task_success", "target_grasp", "pose_caps_satisfied"))
        ):
            raise ValueError("Each endpoint must bind its actual replay state and all inner samples")
        maxima["state"] = max(maxima["state"], float(np.max(np.abs(left_state - right_state))))
        for key in ("q", "qvel", "time"):
            left_inner = np.asarray([row[key] for row in left["actual_inner"]])
            right_inner = np.asarray([row[key] for row in right["actual_inner"]])
            if not np.isfinite(left_inner).all() or not np.isfinite(right_inner).all():
                raise ValueError("Finite actual and recorded inner samples are required")
            maxima[key] = max(maxima[key], float(np.max(np.abs(left_inner - right_inner))))
    if not all(value <= 1e-10 for value in maxima.values()):
        raise ValueError("The cached replay exceeds the registered cross-run physical tolerance")
    current, recorded = actual[-1]["endpoint_receipt"], expected[-1]["endpoint_receipt"]
    captured = capture["current_endpoint"]
    if not (
        capture["frame_index"] == captured["endpoint_step"] == 120
        and captured["sim_state_sha256"] == current["sim_state_sha256"]
        and captured["sim_time"] == current["sim_time"]
        and capture["render_before"] == capture["render_after"]
        and capture["render_before"]["sim_state_sha256"] == current["sim_state_sha256"]
        and capture["render_before"]["sim_time"] == current["sim_time"]
        and capture["future_rgb_frames"] == 0
    ):
        raise ValueError("The current image must bind the unchanged actual replay endpoint")
    return {
        "completed_actions": 40,
        "native_steps": 3000,
        "max_difference": maxima,
        "reference_final_state_sha256": recorded["sim_state_sha256"],
        "replay_final_state_sha256": current["sim_state_sha256"],
        "final_state_bytes_equal": recorded["sim_state_sha256"] == current["sim_state_sha256"],
    }
