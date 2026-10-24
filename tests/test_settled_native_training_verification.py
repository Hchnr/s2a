"""Reject changed native evidence, premature success and noncausal hold targets."""

import copy
import hashlib
import importlib
import json

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.native_training import synchronous_labels


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("verify_settled_native_training")


@pytest.fixture
def measured():
    n, k = 43, 124
    time = 0.25 + np.arange(k, dtype=np.float64) * 0.05
    poses = np.broadcast_to(np.eye(4), (k, 4, 4)).copy()
    poses[:, :3, :3] = Rotation.from_rotvec(np.column_stack((np.zeros((k, 2)), np.arange(k) * 0.001))).as_matrix()
    poses[:, :3, 3] = np.column_stack((np.arange(k) * 0.001, np.zeros(k), np.full(k, 0.5)))
    controller = poses.copy()
    controller[:, 2, 3] += 0.12
    qpos = np.broadcast_to([0.02, -0.02], (k, 2)).copy()
    opening = np.clip((qpos[:, 0] - qpos[:, 1] - 0.001) / 0.079, 0, 1)
    native = np.zeros((n, 7), np.float64)
    native[:, -1] = -1
    commands = np.zeros((n + 80, 7), np.float64)
    commands[:n], commands[n:, -1] = native, native[-1, -1]
    caches = [np.zeros(1)] + [np.clip([i * 0.25, -i * 0.25], -1, 1) for i in range(1, k)]
    offsets = np.r_[0, np.cumsum([len(cache) for cache in caches])].astype(np.int64)
    tool = np.eye(4)
    tool[:3, :3] = Rotation.from_rotvec([0.3, 0.0, 0.0]).as_matrix()
    tool[:3, 3] = [0.01, -0.02, 0.1]
    data = dict(
        endpoint_step=np.arange(k, dtype=np.int64),
        sim_time=time,
        native_success=np.ones(k, np.bool_),
        sim_states=np.column_stack((time, np.zeros((k, 2)))),
        source_position=poses[:, :3, 3].copy(),
        source_quaternion_xyzw=Rotation.from_matrix(poses[:, :3, :3]).as_quat(),
        gripper_qpos=qpos,
        source_pose=poses.copy(),
        actual_source_site_pose=poses.copy(),
        actual_eef_body_pose=poses.copy(),
        physical_source_pose=poses.copy(),
        controller_site_pose=controller,
        opening=opening,
        controller_ee_pos=controller[:, :3, 3].copy(),
        controller_ee_ori_mat=controller[:, :3, :3].copy(),
        controller_goal_pos=controller[:, :3, 3].copy(),
        controller_goal_ori=controller[:, :3, :3].copy(),
        controller_joint_pos=np.zeros((k, 7)),
        controller_joint_vel=np.zeros((k, 7)),
        gripper_current_action_flat=np.concatenate(caches),
        gripper_current_action_offsets=offsets,
        native_commands=native,
        sent_commands=commands,
    )
    data.update(synchronous_labels(poses, opening, tool, time))
    data["controller_goal_pos"][n + 1 :] = controller[n, :3, 3]
    data["controller_goal_ori"][n + 1 :] = controller[n, :3, :3]
    goals = []
    for index in range(n + 80):
        active = index >= n
        goals.append(
            dict(
                step_index=index,
                action=commands[index, :6].tolist(),
                overridden=active,
                set_pos=controller[n, :3, 3].tolist() if active else None,
                set_ori=controller[n, :3, :3].tolist() if active else None,
                actual_goal_pos=data["controller_goal_pos"][index + 1].tolist(),
                actual_goal_ori=data["controller_goal_ori"][index + 1].tolist(),
            )
        )
    return data, goals, n, tool


def test_independent_labels_cover_all_native_and_80_hold_steps(audit, measured):
    data, goals, n, tool = measured
    result = audit.verify_labels(data, n, tool, data["sim_states"][0])
    assert result["position"] < 1e-10 and result["rotation"] < 1e-10
    audit.verify_commands(data, data["native_commands"], n)
    audit.verify_goals(goals, data, n)
    assert audit.anchors_for(n) == [0, 1, 2, 3, n + 40]


@pytest.mark.parametrize(
    "error",
    [
        "opening_time",
        "final_action",
        "final_mask",
        "final_valid",
        "period",
        "source_body",
        "cache_layout",
        "dtype",
        "short",
        "initial",
        "extra_array",
    ],
)
def test_reject_geometry_time_cache_or_terminal_label_errors(audit, measured, error):
    data, _, n, tool = measured
    initial = data["sim_states"][0].copy()
    if error == "opening_time":
        data["action128"][n + 40, 9] += 0.01
    elif error == "final_action":
        data["action128"][-1, 0] = 1
    elif error == "final_mask":
        data["action_mask"][-1, 0] = True
    elif error == "final_valid":
        data["action_valid"][-1] = True
    elif error == "period":
        data["sim_time"][n + 42] += 0.01
    elif error == "source_body":
        data["actual_eef_body_pose"][n, :3, :3] = np.eye(3)
    elif error == "cache_layout":
        data["gripper_current_action_offsets"][n + 1] += 1
    elif error == "dtype":
        data["controller_goal_pos"] = data["controller_goal_pos"].astype(np.float32)
    elif error == "short":
        data["native_success"] = data["native_success"][:-1]
    elif error == "initial":
        initial[1] += 0.01
    else:
        data["hidden_future_pose"] = data["tcp_pose"].copy()
    with pytest.raises(ValueError):
        audit.verify_labels(data, n, tool, initial)


@pytest.mark.parametrize("error", ["native_gripper", "motor_zero", "settling_arm", "verification_gripper"])
def test_gripper_last_native_command_must_continue_in_both_phases(audit, measured, error):
    data, _, n, _ = measured
    index, slot = {
        "native_gripper": (n - 1, 6),
        "motor_zero": (n, 6),
        "settling_arm": (n + 20, 0),
        "verification_gripper": (n + 79, 6),
    }[error]
    data["sent_commands"][index, slot] += 1
    with pytest.raises(ValueError, match="command schedule"):
        audit.verify_commands(data, data["native_commands"], n)


def test_native_prefix_excludes_boundary_outgoing_labels_only(audit, measured):
    data, _, n, _ = measured
    historical = copy.deepcopy(data)
    historical["action128"][n, 0] += 0.01
    historical["delta_pose"][n, 0, 3] += 0.01
    audit.verify_native_prefix(data, historical, n)
    historical["controller_goal_pos"][n, 0] += 1e-12
    with pytest.raises(ValueError, match="native prefix changed: controller_goal_pos"):
        audit.verify_native_prefix(data, historical, n)


@pytest.mark.parametrize(
    "key",
    [
        "native_commands",
        "controller_joint_vel",
        "gripper_current_action_flat",
        "action_mask",
        "action_valid",
        "native_success",
    ],
)
def test_native_prefix_includes_real_controller_cache_and_masks(audit, measured, key):
    data, _, n, _ = measured
    historical = copy.deepcopy(data)
    if historical[key].dtype == np.bool_:
        historical[key][n] = ~historical[key][n]
    else:
        historical[key].flat[0] += 1e-12
    with pytest.raises(ValueError, match=f"native prefix changed: {key}"):
        audit.verify_native_prefix(data, historical, n)


@pytest.mark.parametrize(
    "error",
    ["native_override", "source_instead_site", "settled_instead_native", "late_relock", "cache", "truncated", "order"],
)
def test_latch_uses_only_native_terminal_measured_controller_site(audit, measured, error):
    data, goals, n, _ = measured
    if error == "native_override":
        goals[n - 1]["set_pos"] = [0, 0, 0]
    elif error == "source_instead_site":
        goals[n]["set_pos"] = data["source_pose"][n, :3, 3].tolist()
    elif error == "settled_instead_native":
        goals[n]["set_pos"] = data["controller_site_pose"][n + 40, :3, 3].tolist()
    elif error == "late_relock":
        goals[-1]["set_pos"][0] += 1e-12
    elif error == "cache":
        goals[n + 40]["actual_goal_ori"][0][0] += 1e-12
    elif error == "truncated":
        goals.pop()
    else:
        goals[n]["step_index"] += 1
    with pytest.raises(ValueError):
        audit.verify_goals(goals, data, n)


def test_fixed_settling_prefix_does_not_count_as_final40_success(audit, measured):
    data, _, n, _ = measured
    record = dict(name="example", native_steps=n, task_index=0)
    data["native_success"][: n + 41] = False
    result = audit.summary(record, data)
    assert result["stable_native_success"] is True
    assert result["complete_settling_success"] == [False] * 40
    data["native_success"][n + 41] = False
    result = audit.summary(record, data)
    assert result["stable_native_success"] is False
    assert result["successful_verification_endpoints"] == 39


def test_quality_cannot_drop_failures_or_promote_29_of30(audit):
    rows = [
        dict(name=f"task{task}_{index}", task_index=task, stable_native_success=True)
        for task in range(3)
        for index in range(10)
    ]
    assert audit.quality_summary(rows)["eligible_for_model_training_registration"] is True
    rows[11]["stable_native_success"] = False
    result = audit.quality_summary(rows)
    assert result["stable_native_cases"] == 29
    assert result["stable_native_by_task"] == {"0": 10, "1": 9, "2": 10}
    assert result["eligible_for_model_training_registration"] is False
    with pytest.raises(ValueError, match="all30"):
        audit.quality_summary(rows[:11] + rows[12:])


def test_record_permits_only_relocation_and_settled_fifth_anchor(audit, monkeypatch, tmp_path):
    path = tmp_path / "expert.npy"
    np.save(path, np.zeros((43, 7)))
    historical = dict(
        native_steps=43,
        anchor_frames=[0, 1, 2, 3, 43],
        expert_path="old/expert.npy",
        expert_sha256="gold",
        task="fixed",
        name="source",
    )
    monkeypatch.setattr(audit, "resolve_path", lambda _: path)
    monkeypatch.setattr(audit, "reference", lambda _: dict(path="old/expert.npy", sha256="gold"))
    operational = dict(historical, expert_path=str(path), anchor_frames=[0, 1, 2, 3, 83])
    audit.verify_record_migration(operational, historical)
    operational["task"] = "changed"
    with pytest.raises(ValueError, match="beyond"):
        audit.verify_record_migration(operational, historical)
    operational["task"] = "fixed"
    operational["anchor_frames"][3] = 4
    with pytest.raises(ValueError, match="anchors"):
        audit.verify_record_migration(operational, historical)


def test_original_collection_includes_cached_task0_and_new_tasks1_2(audit, monkeypatch):
    tasks = {
        f"task{i}": dict(complete=True, passed=True, task_index=i, cases=[f"case{i}_{j}" for j in range(10)])
        for i in range(3)
    }
    cases = {f"case{i}_{j}": dict(name=f"case{i}_{j}") for i in range(3) for j in range(10)}
    monkeypatch.setattr(audit, "read", lambda ref: (tasks | cases)[ref])
    collection = dict(
        complete=True, passed=True, cached_task_ref="task0", task_results=[dict(result="task1"), dict(result="task2")]
    )
    result = audit.historical_cases(collection)
    assert len(result) == 30 and result[0]["name"] == "case0_0" and result[-1]["name"] == "case2_9"
    collection.pop("cached_task_ref")
    with pytest.raises(ValueError, match="task roster"):
        audit.historical_cases(collection)


@pytest.mark.parametrize("error", [None, "unreaped", "signal", "different_child", "gpu", "retry"])
def test_score_requires_actual_owned_normal_wait_and_reap(audit, monkeypatch, tmp_path, error):
    launch = dict(
        selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""), child=dict(pid=123, start_ticks=456)
    )
    ended = dict(actual_wait=True, automatic_retry=False, returncode=0)
    release = dict(
        reaped=True,
        returncode=0,
        signal_sent=False,
        original_error=None,
        only_owned_child_signalled=True,
        pid=123,
        start_ticks=456,
    )
    if error == "unreaped":
        release["reaped"] = False
    elif error == "signal":
        release["signal_sent"] = True
    elif error == "different_child":
        release["start_ticks"] += 1
    elif error == "gpu":
        launch["selected_environment"]["CUDA_VISIBLE_DEVICES"] = "2"
    elif error == "retry":
        ended["automatic_retry"] = True
    documents = {"launch.json": launch, "exit.json": ended, "release.json": release}
    monkeypatch.setattr(audit, "reference", lambda path: path)
    monkeypatch.setattr(audit, "read", lambda path: documents[path.name])
    if error is None:
        audit.verify_lifecycle(tmp_path)
    else:
        with pytest.raises(ValueError, match="normal wait/reap/CPU"):
            audit.verify_lifecycle(tmp_path)


@pytest.fixture
def camera(audit, measured, tmp_path):
    data, _, n, _ = measured
    frame = n + 40
    raw = {
        "agentview_image": np.broadcast_to(np.array([30, 70, 110], np.uint8), (256, 256, 3)).copy(),
        "robot0_eye_in_hand_image": np.broadcast_to(np.array([170, 130, 90], np.uint8), (256, 256, 3)).copy(),
    }
    raw.update(
        live_auxiliary_0=raw["agentview_image"][::-1, ::-1], live_left_wrist=raw["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    archive = tmp_path / "cameras.npz"
    np.savez(archive, **raw)
    pixels = np.empty((960, 640, 3), np.uint8)
    pixels[:480], pixels[480:] = [30, 70, 110], [170, 130, 90]
    pixels[478:482] = 0
    png = tmp_path / "canvas.png"
    Image.fromarray(pixels).save(png)
    receipt = dict(sim_time=float(data["sim_time"][frame]), sim_state_sha256="actual-state")
    anchor = dict(
        frame_index=frame,
        current_endpoint=receipt,
        render_before=receipt.copy(),
        render_after=receipt.copy(),
        current_source_pose=data["source_pose"][frame].tolist(),
        current_opening=float(data["opening"][frame]),
        future_rgb_frames=0,
        camera_keys=["agentview_image", "robot0_eye_in_hand_image"],
        camera_roles=["auxiliary_0", "left_wrist"],
        raw_camera_shape=[256, 256, 3],
        raw_camera_dtype="uint8",
        live_flip_axes=[0, 1],
        raw_cameras=audit.reference(archive),
        canvas_png=audit.reference(png),
        canvas_rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
        canvas_shape=[960, 640, 3],
        canvas_dtype="uint8",
    )
    return anchor, data, receipt, pixels


def test_new_fifth_camera_is_bound_to_settled_current_endpoint(audit, camera):
    anchor, data, receipt, _ = camera
    assert audit.verify_canvas(anchor, data, receipt)["maximum_channel_difference"] == 0


@pytest.mark.parametrize("error", ["future", "role", "binding", "raw_changed", "canvas", "seam"])
def test_camera_rejects_hidden_future_changed_native_raw_and_canvas(audit, camera, tmp_path, error):
    anchor, data, receipt, pixels = camera
    historical = None
    if error == "future":
        anchor["future_rgb_frames"] = 1
    elif error == "role":
        anchor["camera_roles"].reverse()
    elif error == "binding":
        anchor["render_after"]["sim_time"] += 0.05
    elif error == "raw_changed":
        historical = copy.deepcopy(anchor)
        raw = audit.array_file(anchor["raw_cameras"])
        raw["agentview_image"][0, 0, 0] += 1
        archive = tmp_path / "historical.npz"
        np.savez(archive, **raw)
        historical["raw_cameras"] = audit.reference(archive)
    else:
        if error == "canvas":
            pixels[0, 0, 0] += 2
        else:
            pixels[479, 0, 0] = 1
        png = tmp_path / "mutated.png"
        Image.fromarray(pixels).save(png)
        anchor["canvas_png"] = audit.reference(png)
        anchor["canvas_rgb_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
    with pytest.raises(ValueError):
        audit.verify_canvas(anchor, data, receipt, historical)


def test_receipts_bind_complete_predicate_and_actual_ragged_cache(audit, measured, tmp_path):
    data, _, _, _ = measured
    receipts = []
    for index, time in enumerate(data["sim_time"]):
        state, cache = data["sim_states"][index], audit.cache_at(data, index)
        receipts.append(
            dict(
                endpoint_step=index,
                sim_time=float(time),
                sim_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                sim_state_dtype=str(state.dtype),
                sim_state_shape=list(state.shape),
                native_success=True,
                gripper_current_action_shape=list(cache.shape),
                gripper_current_action_dtype=str(cache.dtype),
                gripper_current_action_sha256=hashlib.sha256(cache.tobytes()).hexdigest(),
            )
        )
    path = tmp_path / "endpoints.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in receipts))
    case = dict(endpoint_receipts=audit.reference(path))
    assert len(audit.verify_receipts(case, data)) == len(data["sim_time"])
    data["gripper_current_action_flat"][-1] += 1e-12
    with pytest.raises(ValueError, match="cache endpoint receipt"):
        audit.verify_receipts(case, data)
