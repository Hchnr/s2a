"""Identity isolation, physical time binding and current-only export for053."""

import copy
import hashlib

import numpy as np
import pytest

from s2a.synchronous_validation import ROSTER, current_model_input, validate_records


def records():
    values = []
    for i, (task, demo, count) in enumerate(ROSTER):
        state = np.zeros(79)
        state[0] = i
        values.append(
            {
                "name": f"task{task}_validation_{demo}",
                "task_index": task,
                "demo": demo,
                "native_steps": count,
                "task_id": [8, 9, 6][task],
                "initial_state": state.tolist(),
                "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                "anchor_frames": [j * (count - 40) // 3 for j in range(4)],
                "model_xml": "<mujoco/>",
                "model_xml_sha256": hashlib.sha256(b"<mujoco/>").hexdigest(),
            }
        )
    return values


@pytest.mark.parametrize("change", ["alias", "demo", "anchor", "state", "xml", "task", "missing"])
def test_fixed_identity_or_complete_window_violation_rejected(change):
    r = records()
    if change == "alias":
        r[1]["initial_state"] = r[0]["initial_state"]
        r[1]["initial_state_sha256"] = r[0]["initial_state_sha256"]
    elif change == "demo":
        r[0]["demo"] = "demo_20"
    elif change == "anchor":
        r[0]["anchor_frames"][-1] += 1
    elif change == "state":
        r[0]["initial_state"][1] = 1
    elif change == "xml":
        r[0]["model_xml"] = "<mujoco changed='true'/>"
    elif change == "task":
        r[0]["task_id"] = 9
    else:
        r.pop()
    with pytest.raises(ValueError):
        validate_records(r, [])


def test_registered_training_or_fresh_alias_rejected():
    r = records()
    validate_records(r, [])
    with pytest.raises(ValueError):
        validate_records(r, [r[0]["initial_state_sha256"]])


def binding():
    state = np.arange(158, dtype=np.float64).reshape(2, 79)
    data = {
        "sim_states": state,
        "sim_time": np.array([0.25, 0.3]),
        "state128": np.ones((2, 128)),
        "state_mask128": np.ones((2, 128), dtype=bool),
    }
    # Saved trajectories call this field state_mask.
    data["state_mask"] = data.pop("state_mask128")
    anchor = {
        "frame_index": 0,
        "current_endpoint": {"sim_time": 0.25, "sim_state_sha256": hashlib.sha256(state[0].tobytes()).hexdigest()},
        "canvas_png": {"path": "current.png", "sha256": "image"},
    }
    return anchor, data


def test_export_ignores_future_rows_and_never_exports_archive_or_task_truth():
    a, d = binding()
    expected = current_model_input(a, d, task="task", case_name="case")
    d["state128"][1] = -100
    d["sim_states"][1] = -100
    d["action128"] = np.zeros((2, 128))
    d["native_success"] = np.ones(2, dtype=bool)
    assert current_model_input(a, d, task="task", case_name="case") == expected
    assert set(expected) == {
        "name",
        "task",
        "anchor_frame",
        "image_ref",
        "current_time_s",
        "current_sim_state_sha256",
        "measured_state128",
        "state_mask128",
        "initial_sync_certified",
    }


@pytest.mark.parametrize("field", ["sim_time", "sim_state_sha256"])
def test_image_state_clock_or_identity_mismatch_rejected(field):
    a, d = binding()
    a = copy.deepcopy(a)
    a["current_endpoint"][field] = 0.3 if field == "sim_time" else "wrong"
    with pytest.raises(ValueError):
        current_model_input(a, d, task="task", case_name="case")
