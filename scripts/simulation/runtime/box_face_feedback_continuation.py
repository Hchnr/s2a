"""Replay one image-bound reference, then evaluate short face feedback."""

import sys

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import inverse, pose_error
from s2a.references.action_reference import decode_action_targets
from s2a.research.native_runtime.development_continuation import run_continuation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_tabletop_projection_feedback_probe as physical_audit
from scripts.simulation.runtime.box_face_feedback_measurement import BoxFaceInnerRecorder, CurrentBoxFaceGeometry
from scripts.simulation.runtime.early_tabletop_live_continuation import replay_and_bind_image

FORBIDDEN = (
    "s2a.research.joint_control.tracking.box_face_feedback",
    "s2a.research.joint_control.tracking.box_face_reference_projection",
)


def checker_isolation():
    physical_audit.checker_isolation()
    prefix.require(all(name not in sys.modules for name in FORBIDDEN), "The checker cannot import the candidate policy")


def replay_decision(current, saved):
    """Check measured inputs and replay a dispatched command without the policy."""
    decision = saved["decision"]
    np.testing.assert_allclose(current["vertices_tcp"], decision["decision_vertices_tcp"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(current["tcp"], decision["decision_tcp"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(current["slide_axis_tcp"], decision["decision_slide_axis_tcp"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(current["joint_position_m"], decision["decision_joint_position_m"], atol=1e-10, rtol=0)
    return {
        "command": np.asarray(saved["command"], dtype=float),
        "newly_selected": False,
        "selected_target_tcp": decision["selected_target_tcp"],
    }


def execute_face_continuation(env, terminal, observers, receipt, inputs, *, mode, output, producer=None, progress=None):
    """Use only the original branch's already-generated WAM response."""
    if mode != "original":
        raise ValueError("Only the original image-bound branch is registered")
    config = inputs["box_face_feedback"]["config"]
    item = inputs["early_tabletop_live_feedback"]
    old, base = inputs["old"]["old"]["base"]["previous"], inputs["old"]["old"]["base"]
    replayed = replay_and_bind_image(
        env, terminal, observers, receipt, inputs, mode=mode, output=output, progress=progress
    )
    branch, captured = item["branches"][mode], replayed["next_current_image"]
    sample, endpoint = captured["sample"], captured["endpoint"]
    anchor = np.asarray(sample["source_pose"]) @ old["tool"]
    targets, _openings = decode_action_targets(anchor, branch["actions"], horizon=40)
    site_from_tcp = inverse(np.asarray(old["controller"]["source_from_site"])) @ old["tool"]
    sensor = CurrentBoxFaceGeometry(env, site_from_tcp=site_from_tcp, config=config)
    np.testing.assert_allclose(sensor.capture()["tcp"], anchor, atol=1e-9, rtol=0)
    interface = OSCInterface(
        *(old["controller"]["contract"][key] for key in ("input_min", "input_max", "output_min", "output_max"))
    )
    expected = None if producer is None else producer["runtime"]["continuation"]["probe"]["periods"]
    if expected is not None:
        checker_isolation()
        prefix.require(
            len(expected) <= config["new_period_limit"], "Only previously dispatched bounded actions may be replayed"
        )
    previous_command = np.asarray(replayed["probe"]["periods"][-1]["command"], dtype=float)
    recorder = BoxFaceInnerRecorder(env, sensor, periods=config["new_period_limit"])
    selected, below_buffer = [], False

    def choose(q, limits, _jacobian, previous, offset):
        nonlocal below_buffer
        if expected is None and offset == config["new_period_limit"]:
            return {"stop": {"reason": "registered_six_period_limit"}, "newly_selected": False}
        margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
        if expected is None and (below_buffer or margin < config["joint_buffer_rad"]):
            return {
                "stop": {"reason": "observed_joint_buffer_loss", "minimum_current_margin_rad": margin},
                "newly_selected": False,
            }
        current = sensor.capture()
        if expected is None:
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
                previous,
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
        recorder.set_context(120 + offset, selected[-1], current)
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
            start_step=120,
            previous_command=previous_command,
            targets=targets[:8],
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
        "All registered inner steps must be measured",
    )
    if expected is not None:
        checker_isolation()
    return {
        "mode": mode,
        "strategy": "box_face_projection",
        "pre_model_replay": replayed,
        "model_image_ref": branch["image_ref"],
        "model_state_sha256": branch["state_sha256"],
        "model_response_ref": branch["response_ref"],
        "original_targets": targets[:8],
        "probe": probe,
        "inner_samples": recorder.samples,
        "automatic_retry": False,
    }
