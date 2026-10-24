"""Audit a newly registered, complete native + settling dataset from saved bytes.

No simulator, HDF reader, WAM, producer labels or collection routine is used.
Protocol integrity and native task quality are separate conclusions; neither
conclusion authorizes training or reserved-fresh performance evaluation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from run_settled_native_training import publish, read, read_prepared, reference, require, resolve_path


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
    close(np.swapaxes(rotation, -1, -2) @ rotation, np.broadcast_to(np.eye(3), rotation.shape), message)
    close(np.linalg.det(rotation), np.ones(rotation.shape[:-2]), message)
    return value


def cache_at(data, index):
    offsets = data["gripper_current_action_offsets"]
    return data["gripper_current_action_flat"][offsets[index] : offsets[index + 1]]


def anchors_for(n):
    require(type(n) is int and 43 <= n <= 600, "native length changed outside original admission")
    return [int(np.floor(j * (n - 40) / 3)) for j in range(4)] + [n + 40]


def verify_record_migration(operational, historical):
    """Exactly expert relocation and the fifth anchor may change."""
    require(operational.keys() == historical.keys(), "native record field roster changed")
    allowed = {"expert_path", "anchor_frames"}
    require(
        {key: value for key, value in operational.items() if key not in allowed}
        == {key: value for key, value in historical.items() if key not in allowed},
        "native record changed beyond expert_path/anchor_frames",
    )
    n = historical["native_steps"]
    require(
        historical["anchor_frames"] == anchors_for(n)[:4] + [n] and operational["anchor_frames"] == anchors_for(n),
        "original four anchors or new settled fifth anchor changed",
    )
    require(
        str(resolve_path(historical["expert_path"])) == operational["expert_path"],
        "operational expert path differs from explicit resolver",
    )
    require(
        reference(historical["expert_path"])["sha256"] == operational["expert_sha256"],
        "historical native command SHA changed",
    )


def verify_commands(data, native, n):
    expected = np.zeros((n + 80, 7), np.float64)
    expected[:n], expected[n:, -1] = native, native[-1, -1]
    bits(data["native_commands"], native, "selected native expert changed")
    bits(data["sent_commands"], expected, "native/80-step final-gripper command schedule differs")


def verify_labels(data, n, tool, initial):
    """Reconstruct source SITE/BODY, canonical TCP, outgoing labels and masks."""
    k = n + 81
    shapes = dict(
        endpoint_step=(k,),
        sim_time=(k,),
        native_success=(k,),
        sim_states=(k, len(initial)),
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
        sent_commands=(n + 80, 7),
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
    require(
        set(data) == set(shapes) | {"gripper_current_action_flat", "gripper_current_action_offsets"},
        "array roster differs",
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
        require(value.dtype.kind in "biuf" and np.isfinite(value).all(), f"non-numeric/non-finite saved array: {key}")
    bits(data["endpoint_step"], np.arange(k, dtype=np.int64), "full endpoint denominator differs")
    require(data["native_success"].dtype == np.bool_, "native predicate must be boolean")
    close(np.diff(data["sim_time"]), np.full(k - 1, 0.05), "20Hz physical time differs", atol=1e-9)
    bits(data["sim_states"][0], np.asarray(initial, np.float64), "initial simulator state changed or preroll added")
    bits(data["sim_time"], data["sim_states"][:, 0], "physical time is not actual simulator endpoint time")
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
    controller = se3(data["controller_site_pose"], "controller SITE SE3 differs")
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
    close(data["controller_ee_pos"][0], controller[0, :3, 3], "initial controller position not refreshed")
    close(data["controller_ee_ori_mat"][0], controller[0, :3, :3], "initial controller orientation not refreshed")
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
    maxima = dict(position=0.0, rotation=0.0)
    for frame in anchors_for(n):
        reconstructed = tcp[frame].copy()
        for step in range(frame, frame + 40):
            encoded = data["action128"][step]
            relative = np.eye(4)
            relative[:3, 3] = encoded[:3]
            relative[:3, :3] = np.column_stack((encoded[3:6], encoded[6:9], np.cross(encoded[3:6], encoded[6:9])))
            reconstructed = reconstructed @ relative
            close(reconstructed, tcp[step + 1], "40-step anchor reconstruction differs")
            maxima["position"] = max(
                maxima["position"], float(np.max(np.abs(reconstructed[:3, 3] - tcp[step + 1, :3, 3])))
            )
            maxima["rotation"] = max(
                maxima["rotation"], float(np.max(np.abs(reconstructed[:3, :3] - tcp[step + 1, :3, :3])))
            )
        require(data["action_valid"][frame : frame + 40].all(), "anchor has invalid observed transitions")
    return maxima


def verify_native_prefix(data, historical, n):
    """Keep every native endpoint/cache exact; exclude boundary outgoing labels."""
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


def verify_goals(goals, data, n):
    require(len(goals) == n + 80, "full native/settling/verification goal denominator differs")
    terminal = data["controller_site_pose"][n]
    for index, goal in enumerate(goals):
        require(goal["step_index"] == index and goal["overridden"] is (index >= n), "controller goal phase differs")
        close(goal["action"], data["sent_commands"][index, :6], "actual OSC command differs", atol=0)
        close(
            goal["actual_goal_pos"],
            data["controller_goal_pos"][index + 1],
            "actual controller position cache differs",
            atol=0,
        )
        close(
            goal["actual_goal_ori"],
            data["controller_goal_ori"][index + 1],
            "actual controller rotation cache differs",
            atol=0,
        )
        if index >= n:
            for key in ("set_pos", "actual_goal_pos"):
                close(goal[key], terminal[:3, 3], "latch is not measured native terminal SITE position", atol=0)
            for key in ("set_ori", "actual_goal_ori"):
                close(goal[key], terminal[:3, :3], "latch is not measured native terminal SITE rotation", atol=0)
        else:
            require(goal["set_pos"] is goal["set_ori"] is None, "native absolute intervention forbidden")


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


def numpy_canvas(images):
    """Independent half-pixel bilinear cover-crop with a registered one-unit tolerance."""
    require(len(images) == 2, "exactly two current cameras required")
    coords = (np.arange(640, dtype=np.float64) + 0.5) * (256 / 640) - 0.5
    lo = np.floor(coords).astype(int)
    weight = coords - lo
    left, right = np.clip(lo, 0, 255), np.clip(lo + 1, 0, 255)
    result = np.zeros((960, 640, 3), np.uint8)
    for i, image in enumerate(images):
        require(image.shape == (256, 256, 3) and image.dtype == np.uint8, "raw RGB shape/dtype differs")
        pixels = image.astype(np.float64)
        horizontal = pixels[:, left] * (1 - weight)[None, :, None] + pixels[:, right] * weight[None, :, None]
        resized = horizontal[left] * (1 - weight)[:, None, None] + horizontal[right] * weight[:, None, None]
        result[i * 480 : (i + 1) * 480] = np.clip(np.rint(resized[80:560]), 0, 255).astype(np.uint8)
    result[478:482] = 0
    return result


def verify_canvas(anchor, data, receipt, historical=None):
    frame = anchor["frame_index"]
    require(anchor["current_endpoint"] == receipt, "current canvas endpoint receipt differs")
    binding = dict(sim_time=receipt["sim_time"], sim_state_sha256=receipt["sim_state_sha256"])
    require(anchor["render_before"] == anchor["render_after"] == binding, "render changed physical state/time")
    close(anchor["current_source_pose"], data["source_pose"][frame], "canvas measured source binding differs", atol=0)
    close(anchor["current_opening"], data["opening"][frame], "canvas opening binding differs", atol=0)
    require(
        anchor["future_rgb_frames"] == 0
        and anchor["camera_keys"] == ["agentview_image", "robot0_eye_in_hand_image"]
        and anchor["camera_roles"] == ["auxiliary_0", "left_wrist"]
        and anchor["raw_camera_shape"] == [256, 256, 3]
        and anchor["raw_camera_dtype"] == "uint8"
        and anchor["live_flip_axes"] == [0, 1],
        "current camera role/flip/future contract differs",
    )
    raw = array_file(anchor["raw_cameras"])
    keys = {"agentview_image", "robot0_eye_in_hand_image", "live_auxiliary_0", "live_left_wrist"}
    require(set(raw) == keys, "raw current camera roster differs")
    for value in raw.values():
        require(value.shape == (256, 256, 3) and value.dtype == np.uint8, "raw RGB shape/dtype differs")
    for source, live in (("agentview_image", "live_auxiliary_0"), ("robot0_eye_in_hand_image", "live_left_wrist")):
        bits(raw[source][::-1, ::-1], raw[live], "actual current camera flip differs")
    if historical is not None:
        require(historical["frame_index"] == frame, "first four historical anchor frames changed")
        previous = array_file(historical["raw_cameras"])
        require(set(previous) == keys, "historical raw role roster differs")
        for key in keys:
            bits(raw[key], previous[key], "historical native raw RGB changed")
    require(reference(anchor["canvas_png"]["path"]) == anchor["canvas_png"], "canvas PNG digest changed")
    with Image.open(resolve_path(anchor["canvas_png"]["path"])) as image:
        require(image.mode == "RGB" and image.size == (640, 960), "canvas RGB mode/geometry differs")
        pixels = np.asarray(image).copy()
    require(
        anchor["canvas_shape"] == [960, 640, 3]
        and anchor["canvas_dtype"] == "uint8"
        and hashlib.sha256(pixels.tobytes()).hexdigest() == anchor["canvas_rgb_sha256"],
        "canvas pixel digest/layout differs",
    )
    expected = numpy_canvas([raw["live_auxiliary_0"], raw["live_left_wrist"]])
    require(not pixels[478:482].any(), "four-pixel camera seam moved")
    maximum = int(np.max(np.abs(pixels.astype(np.int16) - expected.astype(np.int16))))
    require(maximum <= 1, "independent canvas channel difference exceeds registered one-unit tolerance")
    return dict(frame_index=frame, native_raw_exact=historical is not None, maximum_channel_difference=maximum)


def summary(record, data):
    n = record["native_steps"]
    settling, verification = data["native_success"][n + 1 : n + 41], data["native_success"][n + 41 : n + 81]
    require(
        settling.shape == verification.shape == (40,) and verification.dtype == np.bool_,
        "complete stable-success denominator differs",
    )
    return dict(
        name=record["name"],
        task_index=record["task_index"],
        native_steps=n,
        stable_native_success=bool(verification.all()),
        successful_verification_endpoints=int(verification.sum()),
        complete_settling_success=settling.tolist(),
        complete_verification_success=verification.tolist(),
        complete_native_success_sequence=data["native_success"].tolist(),
        settling_steps=40,
        terminal_verification_steps=40,
        total_hold_steps=80,
    )


def quality_summary(cases):
    require(
        len(cases) == 30
        and [r["task_index"] for r in cases] == [i for i in range(3) for _ in range(10)]
        and len({r["name"] for r in cases}) == 30,
        "native quality requires all30 fixed cases/full denominator",
    )
    stable = sum(row["stable_native_success"] for row in cases)
    return dict(
        total_cases=30,
        stable_native_cases=stable,
        stable_native_by_task={
            str(task): sum(r["stable_native_success"] for r in cases if r["task_index"] == task) for task in range(3)
        },
        failed_native_cases=[row["name"] for row in cases if not row["stable_native_success"]],
        eligible_for_model_training_registration=stable == 30,
    )


def historical_cases(collection):
    """The completed 030 continuation contains cached task0 and new task1/2."""
    require(collection["complete"] is collection["passed"] is True, "original30 collection incomplete")
    tasks = []
    if "cached_task_ref" in collection:
        tasks.append(read(collection["cached_task_ref"]))
    tasks.extend(read(item["result"]) for item in collection["task_results"])
    require([task["task_index"] for task in tasks] == [0, 1, 2], "original30 task roster changed")
    result = []
    for task in tasks:
        require(
            task["complete"] is task["passed"] is True and len(task["cases"]) == 10, "original native task incomplete"
        )
        result.extend(read(ref) for ref in task["cases"])
    return result


def verify_lifecycle(worker):
    launch = read(reference(worker / "launch.json"))
    ended = read(reference(worker / "exit.json"))
    release = read(reference(worker / "ownership/release.json"))
    require(
        ended == dict(actual_wait=True, automatic_retry=False, returncode=0)
        and release["reaped"] is True
        and release["returncode"] == 0
        and release["signal_sent"] is False
        and release["original_error"] is None
        and release["only_owned_child_signalled"] is True
        and launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
        and release["pid"] == launch["child"]["pid"]
        and release["start_ticks"] == launch["child"]["start_ticks"],
        "actual worker normal wait/reap/CPU environment missing",
    )


def verify(args):
    ref = dict(path=str(Path(args.prepared).absolute()), sha256=args.prepared_sha256)
    prepared = read_prepared(ref)
    root, config = Path(prepared["root"]), prepared["config"]
    require(
        config["seed"] == 42
        and config["settling_steps"] == config["terminal_verification_steps"] == 40
        and config["environment_initializations"] == 30
        and config["physics_steps"] == 6180
        and config["current_canvases"] == 150
        and config["model_training_authorized"]
        is config["fresh_performance_authorized"]
        is config["automatic_retry"]
        is False,
        "fixed seed/budget/no-training/no-fresh/no-retry contract changed",
    )
    original = prepared["original_prepared"]
    require(read(config["native_prepared_ref"]) == original, "original complete30 preparation differs")
    require(
        prepared["historical_records"] == original["records"] and len(original["records"]) == 30,
        "historical full30 source roster changed",
    )
    require(
        len(prepared["records"]) == 30
        and [r["task_index"] for r in prepared["records"]] == [i for i in range(3) for _ in range(10)]
        and len({r["name"] for r in prepared["records"]}) == 30,
        "new full30 roster changed",
    )
    for record, historical in zip(prepared["records"], original["records"], strict=True):
        verify_record_migration(record, historical)
    require(sum(r["native_steps"] + 80 for r in prepared["records"]) == 6180, "fixed native+80 budget changed")
    old_verified = read(config["native_verified_ref"])
    require(
        old_verified["passed"] is old_verified["collection_verified"] is True
        and old_verified["native_quality"]["stable_native_cases"] == 28
        and old_verified["native_quality"]["total_cases"] == 30
        and old_verified["eligible_for_model_training_registration"] is False,
        "original030 28/30 False changed",
    )
    old_cases = historical_cases(read(config["native_collection_ref"]))
    require([c["record"] for c in old_cases] == original["records"], "original case/source mapping differs")
    verify_lifecycle(root / "orchestration/worker")
    collected = read(reference(root / "collection/result.json"))
    require(
        collected["passed"] is collected["complete"] is True
        and collected["prepared_ref"] == ref
        and collected["progress"] == dict(environment_initializations=30, physics_steps=6180, current_canvas_images=150)
        and [r["name"] for r in collected["cases"]] == [r["name"] for r in prepared["records"]],
        "complete new fixed30 collection/budget missing",
    )
    require(
        all(
            collected[key] == 0
            for key in (
                "wam_initializations",
                "denoise_calls",
                "fits",
                "fresh_cases",
                "source_hdf5_reads",
                "future_rgb_frames",
            )
        )
        and collected["model_training_authorized"] is collected["fresh_performance_authorized"] is False,
        "new native data collection exceeded CPU/no-model/no-fresh scope",
    )
    rows, canvas_evidence, old_stable = [], [], []
    for row, record, previous in zip(collected["cases"], prepared["records"], old_cases, strict=True):
        case, n = read(row["case"]), record["native_steps"]
        require(
            case["schema_version"] == "s2a.native-training-case.v1"
            and case["complete"] is case["passed"] is True
            and case["prepared_ref"] == ref
            and case["record"] == record
            and case["native_steps"] == n
            and case["physics_steps"] == n + 80
            and case["endpoint_count"] == n + 81
            and case["environment_initializations"] == 1
            and case["current_canvas_images"] == 5
            and case["future_rgb_frames"] == case["source_hdf5_reads"] == 0
            and case["stable_success_last_steps"] == case["settling_steps"] == case["terminal_verification_steps"] == 40
            and case["total_hold_steps"] == 80,
            "complete native/settling/verification case contract differs",
        )
        require(case["controller_contract"] == previous["controller_contract"], "fixed controller changed")
        observation = {
            key: value for key, value in case["observation_contract"].items() if key != "terminal_hold_semantics"
        }
        require(
            observation == previous["observation_contract"]
            and case["observation_contract"]["terminal_hold_semantics"]
            == "hold native-end measured controller SITE pose; continue last native gripper command; first 40 settling, last 40 verification",
            "fixed observation/terminal hold contract changed",
        )
        data, historical = array_file(case["arrays"]), array_file(previous["arrays"])
        require(
            case["array_index"]
            == {key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in data.items()}
            and case["original_dtypes"] == {key: str(value.dtype) for key, value in data.items()},
            "saved array/dtype manifest differs",
        )
        require(
            case["native_command_ref"] == dict(path=record["expert_path"], sha256=record["expert_sha256"]),
            "native expert reference differs",
        )
        require(
            reference(case["native_command_ref"]["path"]) == case["native_command_ref"], "native expert SHA changed"
        )
        native = np.load(resolve_path(record["expert_path"]), allow_pickle=False)
        maxima = verify_labels(data, n, original["tool_source_from_tcp"], record["initial_state"])
        verify_commands(data, native, n)
        verify_native_prefix(data, historical, n)
        receipts = verify_receipts(case, data)
        verify_goals(read(row["goals"]), data, n)
        require(
            [anchor["frame_index"] for anchor in case["anchors"]] == anchors_for(n),
            "five fixed current anchors changed",
        )
        for index, anchor in enumerate(case["anchors"]):
            canvas_evidence.append(
                dict(
                    name=record["name"],
                    **verify_canvas(
                        anchor, data, receipts[anchor["frame_index"]], previous["anchors"][index] if index < 4 else None
                    ),
                )
            )
        item = summary(record, data)
        require(
            case["native_quality_passed"] is item["stable_native_success"],
            "quality flag differs from complete last40 predicates",
        )
        item.update(case_ref=row["case"], maximum_anchor_reconstruction_error=maxima)
        rows.append(item)
        old_stable.append(bool(historical["native_success"][-40:].all()))
    require(
        sum(old_stable) == 28 and len(canvas_evidence) == 150, "original030 quality/full new canvas denominator changed"
    )
    quality = quality_summary(rows)
    read_prepared(ref)
    review = root / "review"
    review.mkdir(exist_ok=False)
    publish(
        review / "verified.json",
        dict(
            passed=True,
            complete=True,
            prepared_ref=ref,
            native_collection_ref=config["native_collection_ref"],
            native_verified_ref=config["native_verified_ref"],
            cases=rows,
            native_quality=quality,
            eligible_for_model_training_registration=quality["eligible_for_model_training_registration"],
            model_training_authorized=False,
            fresh_performance_authorized=False,
            original030_stable_native_cases=28,
            original030_eligible_for_model_training_registration=False,
            native_prefix_exact=True,
            canvas_evidence=canvas_evidence,
            environment_initializations=30,
            physics_steps=6180,
            current_canvas_images=150,
            future_rgb_frames=0,
            new_wam_calls=0,
            new_fit_calls=0,
            source_hdf5_reads=0,
            evidence_level="independent_simulation_saved_endpoint_audit",
            performance_acceptance_completed=False,
            interpretation="New fixed30 native+settling data protocol only; original030 remains28/30 False. Neither native quality nor evidence integrity proves WAM or reserved-fresh task success.",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
