"""Reject causal, control-cache, endpoint and mask errors in the saved-data audit."""

import copy
import importlib
import json

import numpy as np
import pytest

from s2a.artifacts import REPO


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("verify_native_motor_hold_diagnostic")


@pytest.fixture
def measured():
    """Small but complete measured endpoint contract, with two native commands."""
    n, k = 2, 43
    times = 0.25 + np.arange(k, dtype=np.float64) * 0.05
    poses = np.broadcast_to(np.eye(4), (k, 4, 4)).copy()
    poses[:, 0, 3] = np.arange(k) * 0.001
    qpos = np.broadcast_to([0.02, -0.02], (k, 2)).copy()
    opening = np.clip((qpos[:, 0] - qpos[:, 1] - 0.001) / 0.079, 0, 1)
    native = np.zeros((n, 7), np.float64)
    native[:, -1] = -1
    sent = np.zeros((n + 40, 7), np.float64)
    sent[:n] = native
    caches = [np.zeros(1)]
    current = caches[0]
    calls, targets = [], []
    names = ["gripper0_gripper_finger_joint1", "gripper0_gripper_finger_joint2"]
    ranges = np.array([[0.0, 0.04], [-0.04, 0.0]])
    for step, command in enumerate(sent):
        for substep in range(25):
            before = current.copy()
            current = np.clip(before + np.array([-1.0, 1.0]) * 0.01 * np.sign(command[-1:]), -1, 1)
            ctrl = ranges.mean(axis=1) + np.diff(ranges, axis=1)[:, 0] / 2 * current
            calls.append(
                dict(
                    step_index=step,
                    substep_index=substep,
                    sim_time=times[step] + substep * 0.002,
                    input_action=command[-1:].tolist(),
                    cache_before=before.tolist(),
                    cache_after=current.tolist(),
                    actuator_ctrl=ctrl.tolist(),
                    actuator_names=names,
                    actuator_ctrlrange=ranges.tolist(),
                )
            )
        caches.append(current.copy())
        targets.append(
            dict(
                step_index=step,
                sim_time=times[step + 1],
                current_action=current.tolist(),
                actuator_ctrl=ctrl.tolist(),
                actuator_names=names,
                actuator_ctrlrange=ranges.tolist(),
            )
        )
    offsets = np.r_[0, np.cumsum([len(cache) for cache in caches])].astype(np.int64)
    delta = np.linalg.inv(poses[:-1]) @ poses[1:]
    state, action = np.zeros((k, 128)), np.zeros((k, 128))
    state[:, :3] = poses[:, :3, 3]
    state[:, 3:6], state[:, 6:9], state[:, 9] = poses[:, :3, 0], poses[:, :3, 1], opening
    action[:-1, :3], action[:-1, 3:6], action[:-1, 6:9], action[:-1, 9] = (
        delta[:, :3, 3],
        delta[:, :3, 0],
        delta[:, :3, 1],
        opening[1:],
    )
    state_mask = np.zeros((k, 128), np.bool_)
    state_mask[:, :10] = True
    action_mask = state_mask.copy()
    action_mask[-1] = False
    valid = np.ones(k, np.bool_)
    valid[-1] = False
    data = dict(
        endpoint_step=np.arange(k, dtype=np.int64),
        sim_time=times,
        native_success=np.ones(k, np.bool_),
        sim_states=np.column_stack((times, np.zeros((k, 2)))),
        source_position=poses[:, :3, 3].copy(),
        source_quaternion_xyzw=np.tile([0.0, 0.0, 0.0, 1.0], (k, 1)),
        gripper_qpos=qpos,
        source_pose=poses.copy(),
        actual_source_site_pose=poses.copy(),
        actual_eef_body_pose=poses.copy(),
        physical_source_pose=poses.copy(),
        controller_site_pose=poses.copy(),
        opening=opening,
        controller_ee_pos=poses[:, :3, 3].copy(),
        controller_ee_ori_mat=poses[:, :3, :3].copy(),
        controller_goal_pos=poses[:, :3, 3].copy(),
        controller_goal_ori=poses[:, :3, :3].copy(),
        controller_joint_pos=np.zeros((k, 7)),
        controller_joint_vel=np.zeros((k, 7)),
        gripper_current_action_flat=np.concatenate(caches),
        gripper_current_action_offsets=offsets,
        native_commands=native,
        sent_commands=sent,
        tcp_pose=poses.copy(),
        delta_pose=delta,
        state128=state,
        action128=action,
        state_mask=state_mask,
        action_mask=action_mask,
        action_valid=valid,
    )
    return data, targets, calls, n


def test_actual_motor_cache_hold_differs_from_measured_opening(audit, measured):
    data, targets, calls, n = measured
    # The normalized motor latch is a causal controller cache; it is neither
    # measured aperture nor the native binary command (-1).
    assert not np.array_equal(targets[n]["current_action"], [data["opening"][n]] * 2)
    assert targets[n]["current_action"] != [-1, -1]
    evidence = audit.verify_motor_targets(targets, data, n)
    internal = audit.verify_motor_calls(calls, targets, data, n)
    assert evidence["locked_hold_steps"] == 40
    assert internal["held_internal_calls"] == 1000
    audit.verify_commands(data, data["native_commands"], n)
    audit.verify_labels(data, n, np.eye(4))


@pytest.mark.parametrize("error", ["native_reopen", "rescale", "order", "endpoint", "truncated", "outer_hold"])
def test_reject_incorrect_actual_motor_target_or_binding(audit, measured, error):
    data, targets, _, n = measured
    if error == "native_reopen":
        targets[n]["current_action"][0] += 0.01
    elif error == "rescale":
        targets[n]["actuator_ctrl"][0] += 0.001
    elif error == "order":
        targets[n]["actuator_names"].reverse()
    elif error == "endpoint":
        targets[n]["sim_time"] -= 0.05
    elif error == "truncated":
        targets.pop()
    else:
        data["gripper_current_action_flat"][data["gripper_current_action_offsets"][n + 1]] += 0.01
        targets[n]["current_action"][0] += 0.01
        targets[n]["actuator_ctrl"][0] += 0.0002
    with pytest.raises(ValueError):
        audit.verify_motor_targets(targets, data, n)


@pytest.mark.parametrize(
    "error", ["hidden_native_input", "hidden_hold_input", "cache_reset", "clock", "missing", "last_ctrl"]
)
def test_reject_internal_motor_changes_hidden_by_outer_endpoint_logging(audit, measured, error):
    data, targets, calls, n = measured
    if error == "hidden_native_input":
        calls[3]["input_action"] = [1.0]
    elif error == "hidden_hold_input":
        calls[n * 25 + 3]["input_action"] = [-1.0]
    elif error == "cache_reset":
        calls[3]["cache_before"][0] = 0
    elif error == "clock":
        calls[3]["sim_time"] += 0.002
    elif error == "missing":
        calls.pop()
    else:
        calls[24]["actuator_ctrl"][0] += 0.001
    with pytest.raises(ValueError):
        audit.verify_motor_calls(calls, targets, data, n)


def test_native_prefix_excludes_only_boundary_outgoing_labels(audit, measured):
    data, _, _, n = measured
    historical = copy.deepcopy(data)
    # Labels at endpoint n describe the subsequent hold, so their change is
    # expected while endpoint n itself must stay bit-for-bit unchanged.
    historical["action128"][n, 0] += 0.01
    historical["delta_pose"][n, 0, 3] += 0.01
    audit.verify_native_prefix(data, historical, n)
    historical["controller_goal_pos"][n, 0] += 1e-12
    with pytest.raises(ValueError, match="native prefix changed: controller_goal_pos"):
        audit.verify_native_prefix(data, historical, n)


def test_native_cache_prefix_cannot_be_hidden_in_flat_array(audit, measured):
    data, _, _, n = measured
    historical = copy.deepcopy(data)
    historical["gripper_current_action_flat"][historical["gripper_current_action_offsets"][n]] += 1e-12
    with pytest.raises(ValueError, match="native prefix changed: gripper_current_action_flat"):
        audit.verify_native_prefix(data, historical, n)


@pytest.mark.parametrize("error", ["future_opening", "state_mask", "final_action", "final_valid", "period", "frame"])
def test_measured_state_action_and_time_contract(audit, measured, error):
    data, _, _, n = measured
    if error == "future_opening":
        data["action128"][0, 9] += 0.01
    elif error == "state_mask":
        data["state_mask"][-1, 0] = False
    elif error == "final_action":
        data["action128"][-1, 0] = 1
    elif error == "final_valid":
        data["action_valid"][-1] = True
    elif error == "period":
        data["sim_time"][n + 1] += 0.01
    else:
        data["actual_eef_body_pose"][n, :3, :3] = np.diag([-1, -1, 1])
    with pytest.raises(ValueError):
        audit.verify_labels(data, n, np.eye(4))


def test_quality_uses_every_hold_endpoint_and_does_not_count_native_terminal(audit, measured):
    data, _, _, n = measured
    data["native_success"][n] = False
    assert audit.summary("motor_hold", "case", data, n)["stable"]
    data["native_success"][n + 1] = False
    result = audit.summary("motor_hold", "case", data, n)
    assert result["stable"] is False
    assert result["successful_hold_endpoints"] == 39
    assert len(result["complete_hold_success"]) == 40


def test_pose_latch_must_use_measured_native_site_without_future_target(audit, measured):
    data, _, _, n = measured
    historical, goals = [], []
    for index in range(n + 40):
        terminal = data["controller_site_pose"][n]
        latch = index >= n
        goal = dict(
            step_index=index,
            action=data["sent_commands"][index, :6].tolist(),
            overridden=latch,
            set_pos=terminal[:3, 3].tolist() if latch else None,
            set_ori=terminal[:3, :3].tolist() if latch else None,
            actual_goal_pos=terminal[:3, 3].tolist(),
            actual_goal_ori=terminal[:3, :3].tolist(),
        )
        goals.append(goal)
        historical.append(copy.deepcopy(goal))
    audit.verify_goals(goals, historical, data, n, "pose_motor_hold")
    goals[n]["set_pos"] = data["controller_site_pose"][n + 1, :3, 3].tolist()
    with pytest.raises(ValueError, match="measured native terminal SITE"):
        audit.verify_goals(goals, historical, data, n, "pose_motor_hold")


def test_zero_motor_hold_command_does_not_repeat_native_binary_gripper(audit, measured):
    data, _, _, n = measured
    data["sent_commands"][n, -1] = -1
    with pytest.raises(ValueError, match="command schedule"):
        audit.verify_commands(data, data["native_commands"], n)


@pytest.mark.parametrize("error", [None, "wait", "reap", "signal", "cuda", "identity"])
def test_actual_worker_lifecycle_must_prove_owned_cpu_child_normal_release(audit, tmp_path, error):
    worker = tmp_path / "worker"
    (worker / "ownership").mkdir(parents=True)
    exit_record = dict(actual_wait=True, returncode=0)
    release = dict(reaped=True, returncode=0, signal_sent=False, pid=123, start_ticks=456)
    launch = dict(
        selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""), child=dict(pid=123, start_ticks=456)
    )
    if error == "wait":
        exit_record["actual_wait"] = False
    elif error == "reap":
        release["reaped"] = False
    elif error == "signal":
        release["signal_sent"] = True
    elif error == "cuda":
        launch["selected_environment"]["CUDA_VISIBLE_DEVICES"] = "2"
    elif error == "identity":
        release["start_ticks"] += 1
    for name, value in (("exit.json", exit_record), ("ownership/release.json", release), ("launch.json", launch)):
        (worker / name).write_text(json.dumps(value))
    if error is None:
        audit.verify_lifecycle(worker)
    else:
        with pytest.raises(ValueError):
            audit.verify_lifecycle(worker)


def test_expert_relocation_preserves_every_historical_record_field(audit):
    historical = dict(
        name="case",
        expert_path="/share/project/old/expert.npy",
        expert_sha256="f" * 64,
        native_steps=174,
        initial_state=[0.25, 0.1],
        seed=42,
    )
    operational = copy.deepcopy(historical)
    operational["expert_path"] = "/share-mixed/project/new/expert.npy"
    audit.verify_record_migration(operational, historical)
    assert historical["expert_path"] == "/share/project/old/expert.npy"


@pytest.mark.parametrize("error", ["digest", "state", "seed", "missing", "added"])
def test_relocation_cannot_rewrite_source_provenance_or_initial_condition(audit, error):
    historical = dict(
        name="case",
        expert_path="/share/project/old/expert.npy",
        expert_sha256="f" * 64,
        native_steps=174,
        initial_state=[0.25, 0.1],
        seed=42,
    )
    operational = copy.deepcopy(historical)
    operational["expert_path"] = "/share-mixed/project/new/expert.npy"
    if error == "digest":
        operational["expert_sha256"] = "e" * 64
    elif error == "state":
        operational["initial_state"][1] += 0.001
    elif error == "seed":
        operational["seed"] = 43
    elif error == "missing":
        operational.pop("native_steps")
    else:
        operational["alternate_episode"] = True
    with pytest.raises(ValueError):
        audit.verify_record_migration(operational, historical)


def test_historical_array_io_resolves_location_without_rewriting_declared_reference(audit, monkeypatch, tmp_path):
    actual = tmp_path / "relocated.npz"
    expected = np.array([[1.0, 2.0]], np.float64)
    np.savez(actual, measured=expected)
    declared = dict(path="/share/project/old/relocated.npz", sha256="f" * 64)
    observed_paths = []

    def locate(path):
        observed_paths.append(path)
        return actual

    monkeypatch.setattr(audit, "resolve_path", locate)
    monkeypatch.setattr(audit, "reference", lambda path: dict(declared))
    result = audit.array_file(declared)
    assert observed_paths == [declared["path"]]
    assert declared["path"] == "/share/project/old/relocated.npz"
    assert result["measured"].dtype == expected.dtype
    assert result["measured"].tobytes() == expected.tobytes()
