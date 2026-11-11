"""Replay to one verified branch point, then use only its generated reference."""

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import inverse, pose_error
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.native_runtime.streaming_replay import replay_streaming_continuation
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime.tabletop_projection_inner_recording import ProjectionInnerRecorder
from scripts.simulation.runtime.tabletop_projection_measurement import CurrentFingerGeometry


def branch_start(env, sample, item, sensor):
    """Capture the actual branch state and check the source image's anchor."""
    config, source = item["config"], item["source_observation"]
    state = np.asarray(env.sim.get_state().flatten()).copy()
    current = sensor.capture()
    controller, gripper = env.robots[0].controller, env.robots[0].gripper
    prefix.require(
        state_sha256(state) == config["branch_state_sha256"], "The actual branch state must match the source"
    )
    np.testing.assert_allclose(sample["source_pose"], source["source_pose"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(sample["opening"], source["opening"], atol=1e-10, rtol=0)
    return {
        "state_sha256": state_sha256(state),
        "time_s": float(env.sim.data.time),
        "current_tcp": current["tcp"],
        "current_vertices_tcp": current["vertices_tcp"],
        "previous_command": item["previous_command"],
        "controller_goal_position": np.asarray(controller.goal_pos).copy(),
        "controller_goal_rotation": np.asarray(controller.goal_ori).copy(),
        "gripper_motor_target": np.asarray(gripper.current_action).copy(),
    }


def compare_branch_start(actual, baseline):
    """Require matching controller and gripper state before any new dispatch."""
    prefix.require(
        actual["state_sha256"] == baseline["state_sha256"] and actual["time_s"] == baseline["time_s"],
        "Branch state/clock mismatch",
    )
    for key in (
        "current_tcp",
        "current_vertices_tcp",
        "previous_command",
        "controller_goal_position",
        "controller_goal_rotation",
        "gripper_motor_target",
    ):
        np.testing.assert_allclose(actual[key], baseline[key], atol=1e-10, rtol=0)


def execute_continuation(
    env, terminal, observers, receipt, inputs, *, mode, producer=None, baseline_start=None, progress=None
):
    """Use identical replay and a maximum of seven causally measured decisions."""
    item = inputs["tabletop_projection_feedback_probe"]
    config = item["config"]
    old, base = inputs["old"]["old"]["base"]["previous"], inputs["old"]["old"]["base"]
    if mode not in config["modes"] or (mode != "original" and baseline_start is None):
        raise ValueError("A registered mode and baseline anchor are required")
    replay = replay_streaming_continuation(
        env,
        terminal,
        observers,
        receipt,
        producer_stream=item["prefix_stream"],
        target_chunks=item["target_chunks"][:5],
        max_windows=5,
        previous_command=old["commands"][-1],
        tool=old["tool"],
        target_object=base["config"]["target_object"],
        progress=progress,
    )
    prefix.require(
        replay["complete"] is True and replay["control_steps_completed"] == 40,
        "The historical branch prefix must replay",
    )
    sample, endpoint = observers.endpoint(env, 120)
    prefix.require(
        endpoint["endpoint_step"] == 120 and endpoint["sim_state_sha256"] == config["branch_state_sha256"],
        "Exact recorded branch endpoint required",
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
    start = branch_start(env, sample, item, sensor)
    np.testing.assert_allclose(start["current_tcp"], np.asarray(sample["source_pose"]) @ old["tool"], atol=1e-9, rtol=0)
    if baseline_start is not None:
        compare_branch_start(start, baseline_start)
    interface = OSCInterface(
        *(old["controller"]["contract"][key] for key in ("input_min", "input_max", "output_min", "output_max"))
    )
    expected = None if producer is None else producer["runtime"]["continuation"]["probe"]["periods"]
    if expected is not None:
        prefix.require(len(expected) <= 7, "The checker replays at most seven saved decisions")
    recorder = ProjectionInnerRecorder(env, sensor, mode)
    selected = []
    below_buffer = False

    def choose(q, limits, _jacobian, previous, offset):
        nonlocal below_buffer
        if expected is None and offset == 7:
            return {"stop": {"reason": "registered_probe_limit"}, "newly_selected": False}
        margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
        if expected is None and (below_buffer or margin < config["joint_buffer_rad"]):
            return {
                "stop": {"reason": "observed_joint_buffer_loss", "minimum_current_margin_rad": margin},
                "newly_selected": False,
            }
        current = sensor.capture()
        target = np.asarray(item["target_chunks"][5][offset])
        if expected is None:
            from s2a.research.joint_control.tracking.tabletop_projection_feedback import select_tabletop_feedback

            decision = select_tabletop_feedback(
                current["tcp"],
                target,
                site_from_tcp,
                current["vertices_tcp"],
                config["table_pose"],
                config["table_half_size_m"],
                previous,
                interface,
                mode=mode,
                clearance=config["clearance_m"],
                numerical_margin=config["numerical_margin_m"],
                max_translation=config["max_translation_m"],
                command_step_limit=config["command_step_limit"],
            )
            if "stop" in decision:
                return decision
            decision["decision_vertices_tcp"] = current["vertices_tcp"]
            decision["decision_tcp"] = current["tcp"]
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
        recorder.set_context(120 + offset, selected[-1], current["vertices_tcp"])
        return decision

    def completed(actual_env, _command, offset):
        nonlocal below_buffer
        period = receipt["continuation_partial"]["periods"][-1]
        below_buffer = bool(period["inner_below_buffer_samples"])
        measured = sensor.capture()
        error = pose_error(selected[offset], measured["tcp"])
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
        result = run_continuation(
            env,
            {"sample": sample, "endpoint_receipt": endpoint},
            observers,
            receipt,
            start_step=120,
            previous_command=item["previous_command"],
            targets=item["target_chunks"][5],
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
        len(recorder.samples) == 25 * result["steps_completed"] and result["steps_completed"] <= 7,
        "All completed inner samples and the registered period limit are required",
    )
    return {
        "mode": mode,
        "history_replay": replay,
        "branch_start": start,
        "probe": result,
        "inner_samples": recorder.samples,
        "automatic_retry": False,
    }
