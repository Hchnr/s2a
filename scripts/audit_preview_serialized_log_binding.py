"""221 reconstructed command/history log admission; recorded states, no new physics."""

import copy
import json

import numpy as np
from audit_wam_position import Evidence
from executed_residual_independent import ExecutedResidualReplay
from preview_arm_log_replay_serialized import PreviewArmLogReplay

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish
from s2a.executed_command_residual import ExecutedCommandResidualOSC
from s2a.joint_wrench_condition_forward import nominal
from s2a.osc import OSCInterface
from s2a.preview_translation_runtime import plan_current_translation
from s2a.runtime_response_history import RuntimeResponseHistory


def compare(left, right):
    assert set(left) == set(right)
    for key in right:
        if isinstance(right[key], np.ndarray):
            np.testing.assert_allclose(left[key], right[key], atol=1e-9, rtol=0)
        else:
            assert left[key] == right[key]


def main():
    root = REPO / "outputs/preview_serialized_log_binding_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "scripts/preview_arm_log_replay_serialized.py",
        "scripts/serialized_rate_preview_independent.py",
        "tests/test_serialized_preview_audit.py",
        "plans/v0.0.1/222_preview_serialization_recovery_20261026.md",
        "scripts/preview_acceptance_checker_serialized.py",
        "src/s2a/preview_control_simulation.py",
        "scripts/rate_preview_independent.py",
        "scripts/guarded_loading_acceptance_checker.py",
        "scripts/executed_residual_independent.py",
        "src/s2a/executed_command_residual.py",
        "src/s2a/runtime_response_history.py",
        "plans/v0.0.1/221_preview_physical_execution_20261026.md",
    ):
        e.read(path)
    registration = e.json("configs/preview_control_20261026.json")
    for key in ("preview_runtime_ref", "preview_qp_ref", "arm_forward_parameter_ref"):
        ref = registration[key]
        e.check(ref["path"], ref["sha256"])
    proof = e.json(registration["preview_runtime_ref"]["path"])
    assert proof["complete"] and proof["online_input_and_reference_binding"] and proof["rows"] == 164
    for path, signature in proof["source_sha256"].items():
        e.check(path, signature)
    physical = e.json(registration["arm_forward_parameter_ref"]["path"])
    prior = REPO / "outputs/qualified_inverse_guarded_loading_20261025"
    closure = e.json(prior / "closure_verified.json")
    assert closure["complete"] and closure["owner_normal_wait_verified"]
    for path, signature in closure["source_sha256"].items():
        e.check(path, signature)
    case = next((prior / "case00/simulation").glob("*/*"))
    controller, chunks = e.json(case / "controller.json"), e.json(case / "chunks.json")
    rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
    collected = e.json(prior / "case00/collection_result.json")["arrays_ref"]
    e.check(collected["path"], collected["sha256"])
    with np.load(collected["path"], allow_pickle=False) as archive:
        poses = archive["controller_site_pose"].copy()
    config = e.json(prior / "case00/execution_config.json")
    template = e.json(config["policy_template"])
    tool = np.asarray(e.json(template["tool_contract"])["source_from_tcp"])
    mapping = np.linalg.inv(tool) @ np.asarray(controller["source_from_site"])
    contract = controller["contract"]
    interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    converter = ExecutedCommandResidualOSC(
        interface, contract["kp"], contract["kd"], 0.05, compensate=True, inverse_weight=0.5, max_command_rate=5
    )
    source_replay = ExecutedResidualReplay(
        interface, contract["kp"], contract["kd"], 0.05, compensate=True, inverse_weight=0.5, max_command_rate=5
    )
    replay = PreviewArmLogReplay(physical, contract, controller["source_from_site"])
    history = RuntimeResponseHistory()
    previous, rejected = np.zeros(7), []
    for index, old in enumerate(rows):
        row = copy.deepcopy(old)
        data = {key: np.asarray(value).copy() for key, value in row["qualified_forward_input"].items()}
        data["x_previous_command"] = previous[None].copy()
        data = history.attach(data)
        chunk = chunks[row["chunk_index"]]
        cursor = index - chunk["step"]
        generated, anchor = (
            np.asarray(chunk["effective_targets_tcp"]) @ mapping,
            np.asarray(chunk["anchor_tcp"]) @ mapping,
        )
        measured, desired, twist = (
            np.asarray(row[key]) for key in ("measured_source", "desired_source", "measured_twist")
        )
        converted = converter.convert(
            measured, desired, twist, desired_linear_velocity=np.asarray(row["linear_velocity_reference"]["velocity"])
        )
        arm = source_replay.convert(
            measured, desired, twist, desired_linear_velocity=row["linear_velocity_reference"]["velocity"]
        )
        np.testing.assert_allclose(converted["command"], arm["command"], rtol=0, atol=1e-9)
        compare(converted["response_residual"], arm["response_residual"])
        solved = plan_current_translation(data, generated, cursor, anchor, contract)
        command = np.r_[converted["command"], old["command"][-1]]
        command[:3] = solved["command"]
        compare(
            converter.commit_executed(measured, twist, command[:6]),
            source_replay.commit_executed(measured, twist, command[:6]),
        )
        data["action_command"][0] = command
        data["action_goal_delta"][0, :3] = solved["physical_delta"]
        prediction = nominal(data, physical)[0]
        row.update(
            qualified_forward_input=data,
            command=command,
            translation_preview=solved,
            prediction_kind="219 physical preview;150 completed innovation held;not measured future",
            response_residual=dict(innovation=data["x_response_residual"][0].copy(), completed_periods=index),
            nominal_predicted_motion=prediction[:6],
            nominal_predicted_linear_velocity=prediction[6:9] / 0.05,
            predicted_linear_velocity=solved["predicted_velocities"][0],
            rate_limited=bool(np.any(np.isclose(np.abs(command[:3] - previous[:3]), 0.25, atol=1e-10))),
        )
        arguments = (arm, row["desired_source"], row["site_velocity_reference"], generated, cursor, anchor)
        if index == 0:
            paths = [
                ("command",),
                ("translation_preview", "physical_delta"),
                ("translation_preview", "planned_commands"),
                ("translation_preview", "predicted_positions"),
                ("translation_preview", "reference_positions"),
                ("translation_preview", "reference_velocities"),
                ("translation_preview", "completed_innovation"),
                ("qualified_forward_input", "x_previous_command"),
                ("qualified_forward_input", "x_response_residual"),
                ("response_residual", "innovation"),
            ]
            for path in paths:
                broken = copy.deepcopy(row)
                obj = broken
                for key in path[:-1]:
                    obj = obj[key]
                changed = np.asarray(obj[path[-1]], dtype=float).copy()
                changed.flat[0] += 0.02
                obj[path[-1]] = changed
                try:
                    copy.deepcopy(replay).verify(
                        json.loads(json.dumps(broken, default=lambda value: value.tolist(), allow_nan=False)),
                        *arguments,
                    )
                except AssertionError:
                    rejected.append("/".join(path))
                else:
                    raise AssertionError(f"Corrupt preview log accepted: {path}")
        replay.verify(json.loads(json.dumps(row, default=lambda value: value.tolist(), allow_nan=False)), *arguments)
        if index + 1 < len(rows):
            history.observe_completed(data, poses[index + 1], rows[index + 1]["measured_twist"], physical)
        previous = command.copy()
    assert replay.steps == 164 and len(rejected) == 10
    e.recheck()
    root.mkdir()
    publish(
        root / "independent_verified.json",
        dict(
            complete=True,
            rows=164,
            independent_preview_log_and_history=True,
            source_commit_cycles=164,
            json_roundtrip_rows=164,
            corruptions_rejected=rejected,
            source_sha256=e.hashes,
            physics_steps=0,
            evidence_level="recorded states/reconstructed command and history;no counterfactual physical dynamics",
            control_qualified=False,
        ),
    )
    print("221 independent preview log/history/SOURCE commits verified164rows;10 corruptions rejected;no physics")


if __name__ == "__main__":
    main()
