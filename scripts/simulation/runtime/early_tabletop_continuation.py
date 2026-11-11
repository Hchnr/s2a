"""Run two early feedback periods and bind the resulting current image."""

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import inverse, pose_error
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.native_runtime.streaming_replay import replay_streaming_continuation
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime.tabletop_projection_continuation import compare_branch_start
from scripts.simulation.runtime.tabletop_projection_inner_recording import ProjectionInnerRecorder
from scripts.simulation.runtime.tabletop_projection_measurement import CurrentFingerGeometry


def capture_current_image(env, observers, *, step, directory):
    """Bind one rendered image to the actual unchanged completed endpoint."""
    sample, endpoint = observers.endpoint(env, step)
    before = state_sha256(np.asarray(env.sim.get_state().flatten()))
    prefix.require(
        endpoint["endpoint_step"] == step and endpoint["sim_state_sha256"] == before,
        "The image must start from the actual endpoint",
    )
    capture = observers.render_anchor(env, step, sample, endpoint, directory)
    after = state_sha256(np.asarray(env.sim.get_state().flatten()))
    binding = {"sim_time": endpoint["sim_time"], "sim_state_sha256": before}
    prefix.require(
        after == before
        and capture["frame_index"] == step
        and capture["current_endpoint"] == endpoint
        and capture["render_before"] == capture["render_after"] == binding
        and capture["future_rgb_frames"] == 0
        and capture["camera_keys"] == ["agentview_image", "robot0_eye_in_hand_image"]
        and capture["canvas_shape"] == [960, 640, 3]
        and capture["canvas_dtype"] == "uint8",
        "The captured current cameras must not change state or include future frames",
    )
    prefix.checked(capture["canvas_png"])
    return {"sample": sample, "endpoint": endpoint, "capture": capture}


def early_branch_start(env, item, sensor):
    state = np.asarray(env.sim.get_state().flatten()).copy()
    prefix.require(state_sha256(state) == item["config"]["branch_state_sha256"], "The exact step118 state is required")
    current = sensor.capture()
    controller, gripper = env.robots[0].controller, env.robots[0].gripper
    return {
        "state_sha256": state_sha256(state),
        "time_s": float(env.sim.data.time),
        "current_tcp": current["tcp"],
        "current_vertices_tcp": current["vertices_tcp"],
        "previous_command": item["initial_command"],
        "controller_goal_position": np.asarray(controller.goal_pos).copy(),
        "controller_goal_rotation": np.asarray(controller.goal_ori).copy(),
        "gripper_motor_target": np.asarray(gripper.current_action).copy(),
    }


def execute_early_continuation(
    env, terminal, observers, receipt, inputs, *, mode, output, producer=None, baseline_start=None, progress=None
):
    item = inputs["early_tabletop_intervention"]
    previous = inputs["tabletop_projection_feedback_probe"]
    config = item["config"]
    old, base = inputs["old"]["old"]["base"]["previous"], inputs["old"]["old"]["base"]
    if mode not in config["modes"] or (mode != "original" and baseline_start is None):
        raise ValueError("Registered comparison mode and common baseline are required")
    replay = replay_streaming_continuation(
        env,
        terminal,
        observers,
        receipt,
        producer_stream=item["prefix_stream"],
        target_chunks=previous["target_chunks"][:5],
        max_windows=5,
        previous_command=old["commands"][-1],
        tool=old["tool"],
        target_object=base["config"]["target_object"],
        progress=progress,
    )
    prefix.require(
        replay["complete"] is True and replay["control_steps_completed"] == 38,
        "The complete pre-branch history is required",
    )
    sample, endpoint = observers.endpoint(env, 118)
    prefix.require(
        endpoint["endpoint_step"] == 118 and endpoint["sim_state_sha256"] == config["branch_state_sha256"],
        "Exact early branch endpoint required",
    )
    site_from_tcp = inverse(np.asarray(old["controller"]["source_from_site"])) @ old["tool"]
    sensor = CurrentFingerGeometry(
        env,
        site_from_tcp=site_from_tcp,
        table_pose=config["table_pose"],
        table_half_size=config["table_half_size_m"],
        table_name=config["table_geom"],
        finger_name=config["finger_geom"],
    )
    start = early_branch_start(env, item, sensor)
    np.testing.assert_allclose(start["current_tcp"], np.asarray(sample["source_pose"]) @ old["tool"], atol=1e-9, rtol=0)
    if baseline_start is not None:
        compare_branch_start(start, baseline_start)
    interface = OSCInterface(
        *(old["controller"]["contract"][key] for key in ("input_min", "input_max", "output_min", "output_max"))
    )
    expected = None if producer is None else producer["runtime"]["continuation"]["probe"]["periods"]
    if expected is not None:
        prefix.require(len(expected) <= 2, "The checker replays at most two saved actions")
    recorder = ProjectionInnerRecorder(env, sensor, mode, first_step=118, periods=2)
    selected = []
    below_buffer = False

    def choose(q, limits, _jacobian, previous_command, offset):
        nonlocal below_buffer
        if expected is None and offset == 2:
            return {"stop": {"reason": "registered_early_window_limit"}, "newly_selected": False}
        margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
        if expected is None and (below_buffer or margin < config["joint_buffer_rad"]):
            return {
                "stop": {"reason": "observed_joint_buffer_loss", "minimum_current_margin_rad": margin},
                "newly_selected": False,
            }
        current = sensor.capture()
        target = np.asarray(item["source_reference"][6 + offset])
        if expected is None:
            from s2a.research.joint_control.tracking.tabletop_projection_feedback import select_tabletop_feedback

            decision = select_tabletop_feedback(
                current["tcp"],
                target,
                site_from_tcp,
                current["vertices_tcp"],
                config["table_pose"],
                config["table_half_size_m"],
                previous_command,
                interface,
                mode=mode,
                clearance=config["clearance_m"],
                numerical_margin=config["numerical_margin_m"],
                max_translation=config["max_translation_m"],
                command_step_limit=config["command_step_limit"],
            )
            if "stop" in decision:
                return decision
            decision["decision_vertices_tcp"], decision["decision_tcp"] = current["vertices_tcp"], current["tcp"]
        else:
            saved = expected[offset]
            np.testing.assert_allclose(
                current["vertices_tcp"], saved["decision"]["decision_vertices_tcp"], atol=1e-10, rtol=0
            )
            np.testing.assert_allclose(current["tcp"], saved["decision"]["decision_tcp"], atol=1e-10, rtol=0)
            decision = {
                "command": np.asarray(saved["command"], dtype=float),
                "newly_selected": False,
                "selected_target_tcp": saved["decision"]["selected_target_tcp"],
            }
        selected.append(np.asarray(decision["selected_target_tcp"], dtype=float).copy())
        recorder.set_context(118 + offset, selected[-1], current["vertices_tcp"])
        return decision

    def completed(actual_env, _command, offset):
        nonlocal below_buffer
        period = receipt["continuation_partial"]["periods"][-1]
        below_buffer = bool(period["inner_below_buffer_samples"])
        current = sensor.capture()
        error = pose_error(selected[offset], current["tcp"])
        return {
            "projected_target_tcp_error": error,
            "projected_pose_caps_satisfied": bool(
                np.linalg.norm(error[:3]) <= 0.01 and np.linalg.norm(error[3:]) <= 0.05
            ),
            "current_gripper_qpos": np.asarray(actual_env.sim.data.qpos)[
                actual_env.robots[0]._ref_gripper_joint_pos_indexes
            ].copy(),
        }

    env.sim.step = recorder.step
    try:
        probe = run_continuation(
            env,
            {"sample": sample, "endpoint_receipt": endpoint},
            observers,
            receipt,
            start_step=118,
            previous_command=item["initial_command"],
            targets=item["source_reference"][6:14],
            tool=old["tool"],
            target_object=base["config"]["target_object"],
            choose_action=choose,
            stop_on_outside=True,
            expected_periods=expected,
            on_completed=completed,
            progress=progress,
            continuation_limit=8,
        )
    finally:
        env.sim.step = recorder.original
    prefix.require(
        len(recorder.samples) == 25 * probe["steps_completed"] and probe["steps_completed"] <= 2,
        "The early-period sample and action budget must hold",
    )
    capture = (
        None if probe["steps_completed"] != 2 else capture_current_image(env, observers, step=120, directory=output)
    )
    return {
        "mode": mode,
        "history_replay": replay,
        "branch_start": start,
        "probe": probe,
        "inner_samples": recorder.samples,
        "next_current_image": capture,
        "automatic_retry": False,
    }
