"""Independently audit the complete pose × gripper-motor hold diagnostic.

This checker reads saved arrays and receipts only. It imports no collection or
simulation algorithm; numerical reconstruction uses NumPy and saved contracts.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from run_native_motor_hold_diagnostic import publish, read, read_prepared, reference, require, resolve_path

METHODS = ("baseline", "latched_pose", "motor_hold", "pose_motor_hold")
NEW_METHODS = METHODS[2:]
POSE_METHOD = dict(motor_hold="baseline", pose_motor_hold="latched_pose")


def bits(actual, expected, message):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        actual.dtype == expected.dtype and actual.shape == expected.shape and actual.tobytes() == expected.tobytes(),
        message,
    )


def close(actual, expected, message, atol=1e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        actual.shape == expected.shape
        and np.isfinite(actual).all()
        and np.isfinite(expected).all()
        and np.allclose(actual, expected, atol=atol, rtol=0),
        message,
    )


def array_file(ref):
    require(reference(ref["path"]) == ref, "array digest changed")
    with np.load(resolve_path(ref["path"]), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def se3(value, message):
    value = np.asarray(value)
    require(value.shape[-2:] == (4, 4) and np.isfinite(value).all(), message)
    close(value[..., 3, :], np.broadcast_to([0.0, 0.0, 0.0, 1.0], value[..., 3, :].shape), message)
    rotation = value[..., :3, :3]
    close(
        np.swapaxes(rotation, -1, -2) @ rotation,
        np.broadcast_to(np.eye(3), rotation.shape),
        message,
    )
    close(np.linalg.det(rotation), np.ones(rotation.shape[:-2]), message)
    return value


def cache_at(data, index):
    offsets = data["gripper_current_action_offsets"]
    return data["gripper_current_action_flat"][offsets[index] : offsets[index + 1]]


def verify_labels(data, n, tool):
    """Reconstruct measured state and outgoing labels, including the last mask."""
    k = n + 41
    shapes = dict(
        endpoint_step=(k,),
        sim_time=(k,),
        native_success=(k,),
        source_position=(k, 3),
        source_quaternion_xyzw=(k, 4),
        gripper_qpos=(k, 2),
        source_pose=(k, 4, 4),
        actual_source_site_pose=(k, 4, 4),
        actual_eef_body_pose=(k, 4, 4),
        physical_source_pose=(k, 4, 4),
        controller_site_pose=(k, 4, 4),
        tcp_pose=(k, 4, 4),
        opening=(k,),
        native_commands=(n, 7),
        sent_commands=(n + 40, 7),
        controller_ee_pos=(k, 3),
        controller_ee_ori_mat=(k, 3, 3),
        controller_goal_pos=(k, 3),
        controller_goal_ori=(k, 3, 3),
        controller_joint_pos=(k, 7),
        controller_joint_vel=(k, 7),
        delta_pose=(k - 1, 4, 4),
        state128=(k, 128),
        action128=(k, 128),
        state_mask=(k, 128),
        action_mask=(k, 128),
        action_valid=(k,),
    )
    for key, shape in shapes.items():
        require(data[key].shape == shape, f"saved array shape differs: {key}")
        if key not in (
            "endpoint_step",
            "native_success",
            "native_commands",
            "state_mask",
            "action_mask",
            "action_valid",
        ):
            require(data[key].dtype == np.float64, f"saved F64 precision differs: {key}")
    for key, value in data.items():
        require(np.isfinite(value).all(), f"non-finite saved array: {key}")
    bits(data["endpoint_step"], np.arange(k, dtype=np.int64), "full endpoint denominator differs")
    require(data["native_success"].dtype == np.bool_, "native predicate must be boolean")
    close(np.diff(data["sim_time"]), np.full(k - 1, 0.05), "20Hz physical time differs", atol=1e-9)
    require(data["sim_states"].ndim == 2 and data["sim_states"].shape[0] == k, "full simulator state missing")
    require(data["sim_states"].dtype == np.float64, "simulator state lost F64 precision")
    bits(data["sim_time"], data["sim_states"][:, 0], "physical time is not simulator endpoint time")
    flat, offsets = data["gripper_current_action_flat"], data["gripper_current_action_offsets"]
    require(
        flat.ndim == 1
        and flat.dtype == np.float64
        and offsets.shape == (k + 1,)
        and offsets.dtype == np.int64
        and offsets[0] == 0
        and offsets[-1] == len(flat)
        and np.array_equal(np.diff(offsets), np.r_[1, np.full(k - 1, 2)]),
        "actual gripper endpoint cache layout differs",
    )
    source = se3(data["source_pose"], "observed source SE3 differs")
    site = se3(data["actual_source_site_pose"], "actual source SITE SE3 differs")
    body = se3(data["actual_eef_body_pose"], "actual EEF BODY SE3 differs")
    se3(data["controller_site_pose"], "controller SITE SE3 differs")
    close(source[:, :3, 3], data["source_position"], "observed source position differs")
    close(source[:, :3, 3], site[:, :3, 3], "source position is not actual grip site")
    close(source[:, :3, :3], body[:, :3, :3], "source orientation is not actual EEF BODY")
    q = data["source_quaternion_xyzw"]
    close(np.sum(q * q, axis=1), np.ones(k), "source quaternion norm differs")
    x, y, z, w = q.T
    rotation = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(k, 3, 3)
    close(source[:, :3, :3], rotation, "source quaternion/frame differs")
    physical = body.copy()
    physical[:, :3, 3] = site[:, :3, 3]
    close(data["physical_source_pose"], physical, "physical source SITE/BODY construction differs")
    opening = np.clip((data["gripper_qpos"][:, 0] - data["gripper_qpos"][:, 1] - 0.001) / 0.079, 0, 1)
    close(data["opening"], opening, "measured opening differs")
    tcp = source @ se3(tool, "frozen source-from-TCP differs")
    close(data["tcp_pose"], tcp, "canonical measured TCP differs")
    inverse = np.broadcast_to(np.eye(4), (k - 1, 4, 4)).copy()
    inverse[:, :3, :3] = np.swapaxes(tcp[:-1, :3, :3], -1, -2)
    inverse[:, :3, 3] = -np.einsum("nij,nj->ni", inverse[:, :3, :3], tcp[:-1, :3, 3])
    delta = inverse @ tcp[1:]
    close(data["delta_pose"], delta, "local right-multiply transition differs")
    state = np.zeros((k, 128), np.float64)
    state[:, :3], state[:, 3:6], state[:, 6:9], state[:, 9] = tcp[:, :3, 3], tcp[:, :3, 0], tcp[:, :3, 1], opening
    action = np.zeros_like(state)
    action[:-1, :3], action[:-1, 3:6], action[:-1, 6:9], action[:-1, 9] = (
        delta[:, :3, 3],
        delta[:, :3, 0],
        delta[:, :3, 1],
        opening[1:],
    )
    close(data["state128"], state, "measured state labels differ")
    close(data["action128"], action, "observed outgoing action labels differ")
    state_mask = np.zeros((k, 128), np.bool_)
    state_mask[:, :10] = True
    action_mask = state_mask.copy()
    action_mask[-1] = False
    valid = np.ones(k, np.bool_)
    valid[-1] = False
    for key, expected in (("state_mask", state_mask), ("action_mask", action_mask), ("action_valid", valid)):
        bits(data[key], expected, f"complete terminal/state/action mask differs: {key}")


def verify_native_prefix(data, historical, n):
    """All native endpoints remain exact; boundary outgoing labels are excluded."""
    require(data.keys() == historical.keys(), "native array roster changed")
    for key in data:
        if key == "native_commands":
            actual, gold = data[key], historical[key]
        elif key == "gripper_current_action_flat":
            actual = data[key][: data["gripper_current_action_offsets"][n + 1]]
            gold = historical[key][: historical["gripper_current_action_offsets"][n + 1]]
        elif key == "gripper_current_action_offsets":
            actual, gold = data[key][: n + 2], historical[key][: n + 2]
        elif key in ("sent_commands", "delta_pose", "action128"):
            actual, gold = data[key][:n], historical[key][:n]
        else:
            actual, gold = data[key][: n + 1], historical[key][: n + 1]
        bits(actual, gold, f"native prefix changed: {key}")


def verify_record_migration(operational, historical):
    """Only the expert file's operational location may differ from frozen JSON."""
    require(
        operational.keys() == historical.keys() and "expert_path" in historical, "native record field roster changed"
    )
    require(
        {key: value for key, value in operational.items() if key != "expert_path"}
        == {key: value for key, value in historical.items() if key != "expert_path"},
        "native record changed beyond expert_path relocation",
    )
    require(
        isinstance(operational["expert_path"], str) and isinstance(historical["expert_path"], str),
        "native expert location must remain explicit",
    )


def verify_commands(data, native, n):
    expected = np.zeros((n + 40, 7), np.float64)
    expected[:n] = native
    bits(data["native_commands"], native, "selected native expert changed")
    bits(data["sent_commands"], expected, "native/hold command schedule differs")


def verify_goals(goals, historical, data, n, method):
    require(len(goals) == n + 40, "full controller-goal denominator differs")
    require(goals[:n] == historical[:n], "native controller goals changed")
    terminal = data["controller_site_pose"][n]
    for index, goal in enumerate(goals):
        overridden = method == "pose_motor_hold" and index >= n
        require(goal["step_index"] == index and goal["overridden"] is overridden, "controller goal phase differs")
        close(goal["action"], data["sent_commands"][index, :6], "actual OSC command differs", atol=0)
        if overridden:
            close(goal["set_pos"], terminal[:3, 3], "latch is not measured native terminal SITE", atol=0)
            close(goal["set_ori"], terminal[:3, :3], "latch is not measured native terminal SITE", atol=0)
            close(goal["actual_goal_pos"], terminal[:3, 3], "actual OSC position not latched", atol=0)
            close(goal["actual_goal_ori"], terminal[:3, :3], "actual OSC orientation not latched", atol=0)
        else:
            require(goal["set_pos"] is goal["set_ori"] is None, "unexpected native/baseline absolute intervention")


def verify_motor_targets(targets, data, n):
    """Bind actual motor targets to endpoints and independently reconstruct affine control."""
    require(len(targets) == n + 40, "full motor-target denominator differs")
    terminal = cache_at(data, n)
    require(terminal.shape == (2,), "native terminal cache must contain two actual motor targets")
    expected_names = ["gripper0_gripper_finger_joint1", "gripper0_gripper_finger_joint2"]
    ranges = np.asarray([[0.0, 0.04], [-0.04, 0.0]], np.float64)
    hold_ctrl = None
    maximum = 0.0
    for index, target in enumerate(targets):
        require(target["step_index"] == index, "motor target physical-step order differs")
        close(target["sim_time"], data["sim_time"][index + 1], "motor target endpoint time differs", atol=0)
        require(target["actuator_names"] == expected_names, "actual Panda actuator name/order differs")
        actual_range = np.asarray(target["actuator_ctrlrange"])
        close(actual_range, ranges, "actual Panda actuator control ranges differ", atol=0)
        action, control = np.asarray(target["current_action"]), np.asarray(target["actuator_ctrl"])
        require(action.shape == control.shape == (2,), "actual two-finger motor target shape differs")
        close(action, cache_at(data, index + 1), "motor target differs from actual endpoint cache", atol=0)
        require(np.all(np.abs(action) <= 1), "normalized motor target exceeds range")
        reconstructed = (actual_range[:, 1] + actual_range[:, 0]) / 2
        reconstructed += (actual_range[:, 1] - actual_range[:, 0]) / 2 * action
        close(control, reconstructed, "actual motor control is not affine normalized cache", atol=1e-15)
        maximum = max(maximum, float(np.max(np.abs(control - reconstructed))))
        if index == n - 1:
            hold_ctrl = control.copy()
        if index >= n:
            close(action, terminal, "hold motor target not latched at native terminal cache", atol=0)
            close(control, hold_ctrl, "hold actuator control not latched at native terminal", atol=0)
    return dict(
        native_terminal_current_action=terminal.tolist(),
        native_terminal_actuator_ctrl=hold_ctrl.tolist(),
        maximum_affine_control_error=maximum,
        locked_hold_steps=40,
    )


def verify_motor_calls(calls, targets, data, n):
    """Reconstruct every 500Hz Panda cache update from its actual low-level input."""
    require(len(calls) == 25 * (n + 40), "full internal motor-call denominator differs")
    require(len(targets) == n + 40, "full outer motor-target denominator differs")
    previous = cache_at(data, 0)
    require(previous.shape == (1,), "initial actual native cache shape differs")
    maximum = 0.0
    terminal = cache_at(data, n)
    for ordinal, call in enumerate(calls):
        step, substep = divmod(ordinal, 25)
        require(
            call["step_index"] == step and call["substep_index"] == substep,
            "internal motor-call physical/substep order differs",
        )
        close(
            call["sim_time"],
            data["sim_time"][step] + substep * 0.002,
            "internal motor-call 500Hz clock differs",
            atol=1e-9,
        )
        low_input = np.asarray(call["input_action"])
        close(
            low_input,
            data["sent_commands"][step, -1:],
            "internal motor input differs from actual outer command",
            atol=0,
        )
        before, after = np.asarray(call["cache_before"]), np.asarray(call["cache_after"])
        require(before.shape in ((1,), (2,)) and after.shape == (2,), "internal actual cache shape differs")
        close(before, previous, "internal cache continuity differs", atol=0)
        expected = np.clip(before + np.array([-1.0, 1.0]) * 0.01 * np.sign(low_input), -1, 1)
        close(after, expected, "actual internal Panda sign-speed cache recurrence differs", atol=0)
        ranges = np.asarray(call["actuator_ctrlrange"])
        close(ranges, [[0.0, 0.04], [-0.04, 0.0]], "internal actual Panda actuator ranges differ", atol=0)
        require(
            call["actuator_names"] == ["gripper0_gripper_finger_joint1", "gripper0_gripper_finger_joint2"],
            "internal actual Panda actuator order differs",
        )
        expected_ctrl = (ranges[:, 1] + ranges[:, 0]) / 2 + (ranges[:, 1] - ranges[:, 0]) / 2 * after
        close(call["actuator_ctrl"], expected_ctrl, "internal actuator control is not affine cache", atol=1e-15)
        maximum = max(maximum, float(np.max(np.abs(np.asarray(call["actuator_ctrl"]) - expected_ctrl))))
        if step >= n:
            close(low_input, np.zeros(1), "hold internal gripper command must be zero", atol=0)
            close(after, terminal, "hold internal motor cache not native-terminal latch", atol=0)
        if substep == 24:
            close(after, cache_at(data, step + 1), "last internal cache does not bind physical endpoint", atol=0)
            close(
                after, targets[step]["current_action"], "last internal cache does not bind actual outer target", atol=0
            )
            close(
                call["actuator_ctrl"],
                targets[step]["actuator_ctrl"],
                "last internal ctrl does not bind outer target",
                atol=0,
            )
        previous = after
    return dict(
        internal_motor_calls=len(calls),
        calls_per_physical_step=25,
        internal_frequency_hz=500,
        held_internal_calls=1000,
        maximum_internal_affine_control_error=maximum,
    )


def verify_receipts(case, data):
    ref = case["endpoint_receipts"]
    require(reference(ref["path"]) == ref, "endpoint receipts digest changed")
    receipts = [json.loads(line) for line in resolve_path(ref["path"]).read_text().splitlines()]
    require(len(receipts) == len(data["sim_time"]), "endpoint receipts denominator differs")
    for index, receipt in enumerate(receipts):
        state, cache = data["sim_states"][index], cache_at(data, index)
        expected = dict(
            endpoint_step=index,
            sim_time=float(data["sim_time"][index]),
            sim_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
            sim_state_dtype=str(state.dtype),
            sim_state_shape=list(state.shape),
            native_success=bool(data["native_success"][index]),
            gripper_current_action_shape=list(cache.shape),
            gripper_current_action_dtype=str(cache.dtype),
            gripper_current_action_sha256=hashlib.sha256(cache.tobytes()).hexdigest(),
        )
        require(receipt == expected, "actual predicate/state/cache endpoint receipt differs")
    return receipts


def verify_canvas(actual, historical, paired, data, receipt):
    require(
        actual["frame_index"] == historical["frame_index"] == paired["frame_index"], "native camera schedule differs"
    )
    frame = actual["frame_index"]
    binding = dict(sim_time=receipt["sim_time"], sim_state_sha256=receipt["sim_state_sha256"])
    require(actual["current_endpoint"] == receipt, "current canvas endpoint receipt differs")
    require(actual["render_before"] == actual["render_after"] == binding, "render changed physical state/time")
    close(actual["current_source_pose"], data["source_pose"][frame], "canvas measured source binding differs", atol=0)
    close(actual["current_opening"], data["opening"][frame], "canvas opening binding differs", atol=0)
    require(actual["future_rgb_frames"] == 0, "future image not authorized")
    require(
        actual["camera_keys"] == ["agentview_image", "robot0_eye_in_hand_image"]
        and actual["camera_roles"] == ["auxiliary_0", "left_wrist"]
        and actual["raw_camera_shape"] == [256, 256, 3]
        and actual["raw_camera_dtype"] == "uint8"
        and actual["live_flip_axes"] == [0, 1],
        "current camera role/flip contract differs",
    )
    raws = [array_file(item["raw_cameras"]) for item in (actual, historical, paired)]
    expected_keys = {"agentview_image", "robot0_eye_in_hand_image", "live_auxiliary_0", "live_left_wrist"}
    require(all(set(value) == expected_keys for value in raws), "raw current camera roster differs")
    for key in expected_keys:
        require(raws[0][key].shape == (256, 256, 3) and raws[0][key].dtype == np.uint8, "raw RGB shape/dtype differs")
        bits(raws[0][key], raws[1][key], "historical native raw RGB changed")
        bits(raws[0][key], raws[2][key], "paired current native raw RGB changed")
    for source, live in (("agentview_image", "live_auxiliary_0"), ("robot0_eye_in_hand_image", "live_left_wrist")):
        bits(raws[0][source][::-1, ::-1], raws[0][live], "actual current camera flip differs")
    pixels = []
    for item in (actual, historical, paired):
        require(reference(item["canvas_png"]["path"]) == item["canvas_png"], "canvas PNG digest changed")
        with Image.open(resolve_path(item["canvas_png"]["path"])) as image:
            value = np.asarray(image).copy()
        require(value.shape == (960, 640, 3) and value.dtype == np.uint8, "canvas RGB shape/dtype differs")
        require(hashlib.sha256(value.tobytes()).hexdigest() == item["canvas_rgb_sha256"], "canvas pixel digest differs")
        pixels.append(value)
    bits(pixels[0], pixels[2], "paired current native canvas changed")
    maximum = int(np.max(np.abs(pixels[0].astype(np.int16) - pixels[1].astype(np.int16))))
    require(maximum <= 1, "historical canvas channel difference exceeds registered one-unit tolerance")
    return dict(
        frame_index=frame, raw_exact=True, paired_canvas_exact=True, historical_maximum_channel_difference=maximum
    )


def summary(method, name, data, n):
    success = data["native_success"][n + 1 : n + 41]
    require(success.shape == (40,) and success.dtype == np.bool_, "complete stable-success denominator differs")
    return dict(
        method=method,
        name=name,
        stable=bool(success.all()),
        successful_hold_endpoints=int(success.sum()),
        complete_hold_success=success.tolist(),
        native_steps=n,
        hold_endpoint_count=40,
        pose_latched=method in ("latched_pose", "pose_motor_hold"),
        gripper_motor_latched=method in NEW_METHODS,
        terminal_tcp_displacement_mm=((data["tcp_pose"][-1, :3, 3] - data["tcp_pose"][n, :3, 3]) * 1000).tolist(),
        terminal_opening=[float(data["opening"][n]), float(data["opening"][-1])],
    )


def verify_lifecycle(worker):
    exit_record = json.loads((worker / "exit.json").read_text())
    release = json.loads((worker / "ownership/release.json").read_text())
    launch = json.loads((worker / "launch.json").read_text())
    require(
        exit_record["actual_wait"] is True
        and exit_record["returncode"] == 0
        and release["reaped"] is True
        and release["returncode"] == 0
        and release["signal_sent"] is False
        and launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        "actual worker normal wait/reap/CPU environment missing",
    )
    require(
        release["pid"] == launch["child"]["pid"] and release["start_ticks"] == launch["child"]["start_ticks"],
        "worker ownership identity differs",
    )


def verify(args):
    ref = dict(path=str(Path(args.prepared).resolve()), sha256=args.prepared_sha256)
    prepared = read_prepared(ref)
    root, config = Path(prepared["root"]), prepared["config"]
    require(config["methods"] == list(NEW_METHODS), "fixed two-method motor intervention changed")
    require(config["training_authorized"] is config["fresh_authorized"] is False, "training/fresh not authorized")
    pose_prepared = read(config["pose_prepared_ref"])
    require(prepared["pose_prepared"] == pose_prepared, "cached pose preparation changed")
    old = read(config["pose_collection_ref"])
    old_verified = read(config["pose_verified_ref"])
    require(
        old["passed"]
        and old["complete"]
        and old_verified["passed"]
        and old_verified["complete"]
        and old["prepared_ref"] == old_verified["prepared_ref"] == config["pose_prepared_ref"],
        "cached pose comparison completeness/provenance differs",
    )
    require(
        Path(config["pose_verified_ref"]["path"]) == Path(pose_prepared["root"]) / "canvas_addendum/verified.json",
        "historical quality must use frozen canvas addendum",
    )
    verify_lifecycle(root / "orchestration/worker")
    collected = json.loads((root / "collection/result.json").read_text())
    require(collected["passed"] and collected["complete"] and collected["prepared_ref"] == ref, "worker incomplete")
    require(
        collected["progress"] == dict(environment_initializations=8, physics_steps=1648, current_canvas_images=40),
        "full fixed intervention budget missing",
    )
    records = prepared["records"]
    require(
        prepared["historical_records"] == pose_prepared["records"]
        and len(records) == 4
        and len({r["name"] for r in records}) == 4,
        "fixed four native records changed",
    )
    for operational_record, historical_record in zip(records, prepared["historical_records"], strict=True):
        verify_record_migration(operational_record, historical_record)
        require(
            str(resolve_path(historical_record["expert_path"])) == operational_record["expert_path"],
            "operational expert path differs from explicit historical resolver",
        )
        require(
            reference(historical_record["expert_path"])["sha256"] == operational_record["expert_sha256"],
            "historical expert digest changed during relocation",
        )
    operational_records = {record["name"]: record for record in records}
    expected_old = [(method, r["name"]) for method in METHODS[:2] for r in records]
    expected_new = [(method, r["name"]) for method in NEW_METHODS for r in records]
    require([(r["method"], r["name"]) for r in old["cases"]] == expected_old, "cached eight-case roster differs")
    require(
        [(r["method"], r["name"]) for r in old_verified["cases"]] == expected_old, "cached quality denominator differs"
    )
    require([(r["method"], r["name"]) for r in collected["cases"]] == expected_new, "new eight-case roster differs")
    old_rows = {(r["method"], r["name"]): r for r in old["cases"]}
    old_cases = {key: read(row["case"]) for key, row in old_rows.items()}
    current = {(r["method"], r["name"]): read(r["case"]) for r in collected["cases"]}
    tool = pose_prepared["static_prepared"]["native_prepared"]["tool_source_from_tcp"]
    summaries, motor_evidence, canvas_evidence = [], [], []
    for row, reported in zip(old["cases"], old_verified["cases"], strict=True):
        case = old_cases[(row["method"], row["name"])]
        data = array_file(case["arrays"])
        item = summary(row["method"], row["name"], data, case["native_steps"])
        for key in (
            "method",
            "name",
            "stable",
            "successful_hold_endpoints",
            "complete_hold_success",
            "terminal_tcp_displacement_mm",
            "terminal_opening",
        ):
            require(item[key] == reported[key], f"cached quality differs from original arrays: {key}")
        item["cached_case_ref"] = row["case"]
        summaries.append(item)
    for row in collected["cases"]:
        method, name = row["method"], row["name"]
        case = current[(method, name)]
        previous_row = old_rows[(POSE_METHOD[method], name)]
        previous = old_cases[(POSE_METHOD[method], name)]
        data, historical = array_file(case["arrays"]), array_file(previous["arrays"])
        n = previous["native_steps"]
        require(
            case["complete"]
            and case["passed"]
            and case["prepared_ref"] == ref
            and case["record"] == operational_records[name]
            and case["native_steps"] == n
            and case["physics_steps"] == n + 40
            and case["endpoint_count"] == n + 41
            and case["environment_initializations"] == 1
            and case["current_canvas_images"] == 5
            and case["future_rgb_frames"] == case["source_hdf5_reads"] == 0
            and case["stable_success_last_steps"] == 40,
            "complete native case contract differs",
        )
        verify_record_migration(case["record"], previous["record"])
        require(
            case["controller_contract"] == previous["controller_contract"]
            and case["observation_contract"] == previous["observation_contract"],
            "fixed observation/controller changed",
        )
        require(
            case["array_index"]
            == {key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in data.items()},
            "saved array manifest differs",
        )
        require(
            case["original_dtypes"] == {key: str(value.dtype) for key, value in data.items()},
            "saved dtype manifest differs",
        )
        require(
            case["native_command_ref"]
            == dict(path=case["record"]["expert_path"], sha256=case["record"]["expert_sha256"]),
            "native expert reference differs",
        )
        require(
            reference(case["native_command_ref"]["path"]) == case["native_command_ref"], "native expert SHA changed"
        )
        native = np.load(resolve_path(case["native_command_ref"]["path"]), allow_pickle=False)
        verify_labels(data, n, tool)
        verify_commands(data, native, n)
        verify_native_prefix(data, historical, n)
        receipts = verify_receipts(case, data)
        verify_goals(read(row["goals"]), read(previous_row["goals"]), data, n, method)
        motor_targets = read(row["motor_targets"])
        motor_evidence.append(
            dict(
                method=method,
                name=name,
                **verify_motor_targets(motor_targets, data, n),
                **verify_motor_calls(read(row["motor_calls"]), motor_targets, data, n),
            )
        )
        paired_method = "pose_motor_hold" if method == "motor_hold" else "motor_hold"
        paired = current[(paired_method, name)]
        require(
            [anchor["frame_index"] for anchor in case["anchors"]] == case["record"]["anchor_frames"],
            "complete five-anchor roster differs",
        )
        for anchor, old_anchor, paired_anchor in zip(
            case["anchors"], previous["anchors"], paired["anchors"], strict=True
        ):
            canvas_evidence.append(
                dict(
                    method=method,
                    name=name,
                    **verify_canvas(anchor, old_anchor, paired_anchor, data, receipts[anchor["frame_index"]]),
                )
            )
        item = summary(method, name, data, n)
        require(case["native_quality_passed"] is item["stable"], "quality summary differs from all40 native endpoints")
        item["case_ref"] = row["case"]
        summaries.append(item)
    require(len(summaries) == 16 and len(canvas_evidence) == 40, "complete factorial denominator differs")
    grouped = {
        method: dict(
            stable=sum(r["stable"] for r in summaries if r["method"] == method),
            cases=4,
            stable_all=all(r["stable"] for r in summaries if r["method"] == method),
        )
        for method in METHODS
    }
    read_prepared(ref)
    review = root / "review"
    review.mkdir(exist_ok=False)
    publish(
        review / "verified.json",
        dict(
            passed=True,
            complete=True,
            prepared_ref=ref,
            cached_pose_verified_ref=config["pose_verified_ref"],
            cached_pose_collection_ref=config["pose_collection_ref"],
            cases=summaries,
            factorial_summary=grouped,
            native_prefix_exact=True,
            motor_target_evidence=motor_evidence,
            canvas_evidence=canvas_evidence,
            quality_passed=all(grouped[m]["stable_all"] for m in NEW_METHODS),
            training_authorized=False,
            fresh_authorized=False,
            new_model_calls=0,
            new_fit_calls=0,
            evidence_level="independent_simulation_saved_endpoint_audit",
            performance_acceptance_completed=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
