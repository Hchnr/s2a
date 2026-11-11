"""Compare early feedback from one saved state and one prior WAM reference."""

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import inverse, pose_error
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.native_runtime.streaming_replay import replay_streaming_continuation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime.box_face_feedback_continuation import checker_isolation, replay_decision
from scripts.simulation.runtime.box_face_feedback_measurement import BoxFaceInnerRecorder, CurrentBoxFaceGeometry
from scripts.simulation.runtime.early_tabletop_continuation import early_branch_start
from scripts.simulation.runtime.tabletop_projection_continuation import compare_branch_start


def common_stop(q, limits, below_buffer, current_joint, joint_range, *, joint_buffer):
    """Apply the same currently observable arm and finger gates to both modes."""
    q, limits = np.asarray(q, dtype=float), np.asarray(limits, dtype=float)
    interval = np.asarray(joint_range, dtype=float)
    if (
        q.shape != (7,)
        or limits.shape != (7, 2)
        or interval.shape != (2,)
        or not np.isfinite(q).all()
        or not np.isfinite(limits).all()
        or not np.isfinite(interval).all()
        or not np.isfinite(current_joint)
    ):
        raise ValueError("Finite current robot joints and fixed ranges are required")
    margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
    if below_buffer or margin < joint_buffer:
        return {
            "stop": {"reason": "observed_joint_buffer_loss", "minimum_current_margin_rad": margin},
            "newly_selected": False,
        }
    if not interval[0] <= current_joint <= interval[1]:
        return {
            "stop": {"reason": "finger_joint_outside_declared_range", "joint_position_m": float(current_joint)},
            "newly_selected": False,
        }
    return None


def execute_early_box_face(env, terminal, observers, receipt, inputs, *, mode, producer=None, progress=None):
    """Dispatch each branch from current feedback and the fixed step112 reference."""
    item, registered = inputs["early_tabletop_intervention"], inputs["early_box_face_feedback"]
    previous, config = inputs["tabletop_projection_feedback_probe"], registered["config"]
    old, base = inputs["old"]["old"]["base"]["previous"], inputs["old"]["old"]["base"]
    if mode not in config["modes"]:
        raise ValueError("One registered comparison mode is required")
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
        "The full pre-branch history is required",
    )
    sample, endpoint = observers.endpoint(env, 118)
    prefix.require(
        endpoint["endpoint_step"] == 118 and endpoint["sim_state_sha256"] == config["branch_state_sha256"],
        "The exact common branch endpoint is required",
    )
    site_from_tcp = inverse(np.asarray(old["controller"]["source_from_site"])) @ old["tool"]
    sensor = CurrentBoxFaceGeometry(env, site_from_tcp=site_from_tcp, config=config)
    start = early_branch_start(env, item, sensor)
    compare_branch_start(start, registered["baseline_start"])
    np.testing.assert_allclose(start["current_tcp"], np.asarray(sample["source_pose"]) @ old["tool"], atol=1e-9, rtol=0)
    interface = OSCInterface(
        *(old["controller"]["contract"][key] for key in ("input_min", "input_max", "output_min", "output_max"))
    )
    targets = np.asarray(item["source_reference"][6:14], dtype=float)
    expected = None if producer is None else producer["runtime"]["continuation"]["probe"]["periods"]
    if expected is not None:
        checker_isolation()
        prefix.require(
            len(expected) <= config["new_period_limit"], "The checker may replay only the saved bounded actions"
        )
    recorder = BoxFaceInnerRecorder(env, sensor, first_step=118, periods=config["new_period_limit"])
    selected, below_buffer = [], False

    def choose(q, limits, _jacobian, previous_command, offset):
        nonlocal below_buffer
        current = sensor.capture()
        if expected is None:
            stop = common_stop(
                q,
                limits,
                below_buffer,
                current["joint_position_m"],
                sensor.interval,
                joint_buffer=config["joint_buffer_rad"],
            )
            if stop is not None:
                return stop
            if mode == "original":
                from s2a.research.joint_control.tracking.tabletop_projection_feedback import select_tabletop_feedback

                decision = select_tabletop_feedback(
                    current["tcp"],
                    targets[offset],
                    site_from_tcp,
                    current["vertices_tcp"],
                    config["table_pose"],
                    config["table_half_size_m"],
                    previous_command,
                    interface,
                    mode="original",
                    clearance=config["clearance_m"],
                    numerical_margin=config["numerical_margin_m"],
                    max_translation=config["max_translation_m"],
                    command_step_limit=config["command_step_limit"],
                )
            else:
                from s2a.research.joint_control.tracking.box_face_feedback import select_box_face_feedback

                decision = select_box_face_feedback(
                    current["tcp"],
                    targets[offset],
                    site_from_tcp,
                    current["vertices_tcp"],
                    current["slide_axis_tcp"],
                    current["joint_position_m"],
                    sensor.interval,
                    sensor.boxes,
                    previous_command,
                    interface,
                    position_uncertainty=config["position_uncertainty_m"],
                    rotation_uncertainty=config["rotation_uncertainty_rad"],
                    additional_shape_uncertainty=config["additional_shape_uncertainty_m"],
                    clearance=config["clearance_m"],
                    numerical_margin=config["numerical_margin_m"],
                    max_translation=config["max_translation_m"],
                    command_step_limit=config["command_step_limit"],
                )
            if "stop" in decision:
                return decision
            decision.update(
                decision_vertices_tcp=current["vertices_tcp"],
                decision_tcp=current["tcp"],
                decision_slide_axis_tcp=current["slide_axis_tcp"],
                decision_joint_position_m=current["joint_position_m"],
            )
        else:
            decision = replay_decision(current, expected[offset])
        selected.append(np.asarray(decision["selected_target_tcp"], dtype=float).copy())
        recorder.set_context(118 + offset, selected[-1], current)
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
            "current_slide_joint_position_m": measured["joint_position_m"],
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
            targets=targets,
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
        len(recorder.samples) == 25 * probe["steps_completed"]
        and probe["steps_completed"] <= config["new_period_limit"],
        "Every registered inner step must be recorded",
    )
    if expected is not None:
        checker_isolation()
    return {
        "mode": mode,
        "history_replay": replay,
        "branch_start": start,
        "reference_step": 112,
        "reference_image_ref": item["source_capture_ref"],
        "reference_response_ref": item["source_response_ref"],
        "original_targets": targets,
        "probe": probe,
        "inner_samples": recorder.samples,
        "automatic_retry": False,
    }
