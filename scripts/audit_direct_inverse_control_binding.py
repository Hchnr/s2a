"""213 admission: direct commands and completed history on recorded physical states."""

import copy
import json

import numpy as np
from audit_wam_position import Evidence
from direct_inverse_arm_log_replay import DirectInverseArmLogReplay

from s2a.artifacts import REPO
from s2a.direct_arm_inverse_runtime import translation_command
from s2a.durable_stage_supervisor import publish
from s2a.joint_wrench_condition_forward import nominal
from s2a.runtime_response_history import RuntimeResponseHistory


def main():
    root = REPO / "outputs/direct_inverse_control_binding_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "scripts/direct_inverse_arm_log_replay.py",
        "scripts/direct_arm_inverse_independent.py",
        "src/s2a/direct_inverse_control_simulation.py",
        "src/s2a/direct_arm_inverse_runtime.py",
        "src/s2a/runtime_response_history.py",
        "plans/v0.0.1/213_direct_arm_inverse_control_20261026.md",
    ):
        e.read(path)
    registration = e.json("configs/direct_inverse_control_20261026.json")
    for key in ("direct_inverse_parameter_ref", "direct_inverse_runtime_ref", "arm_forward_parameter_ref"):
        ref = registration[key]
        e.check(ref["path"], ref["sha256"])
    inverse = e.json(registration["direct_inverse_parameter_ref"]["path"])
    physical = e.json(registration["arm_forward_parameter_ref"]["path"])
    proof = e.json(registration["direct_inverse_runtime_ref"]["path"])
    assert proof["complete"] and proof["independent_command_verified"] and proof["rows"] == 6180
    for path, signature in proof["source_sha256"].items():
        e.check(path, signature)
    prior = REPO / "outputs/qualified_inverse_guarded_loading_20261025"
    closure = e.json(prior / "closure_verified.json")
    assert closure["complete"] and closure["owner_normal_wait_verified"]
    for path, signature in closure["source_sha256"].items():
        e.check(path, signature)
    case = next((prior / "case00/simulation").glob("*/*"))
    controller = e.json(case / "controller.json")
    rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
    collected = e.json(prior / "case00/collection_result.json")["arrays_ref"]
    e.check(collected["path"], collected["sha256"])
    with np.load(collected["path"], allow_pickle=False) as a:
        poses = a["controller_site_pose"].copy()
    replay = DirectInverseArmLogReplay(physical, controller["contract"], controller["source_from_site"], inverse)
    history = RuntimeResponseHistory()
    previous = np.zeros(7)
    rejected = []
    for index, old in enumerate(rows):
        row = copy.deepcopy(old)
        data = {key: np.asarray(value).copy() for key, value in row["qualified_forward_input"].items()}
        data["x_previous_command"] = previous[None].copy()
        data = history.attach(data)
        command = np.asarray(row["command"]).copy()
        result = translation_command(
            data, np.asarray(row["desired_site"])[None], gripper_command=command[-1:], parameters=inverse
        )
        solved = {
            key: value[0].copy() if isinstance(value[0], np.ndarray) else value[0] for key, value in result.items()
        }
        command[:3] = solved["command"]
        data["action_command"][0] = command
        data["action_goal_delta"][0, :3] = solved["physical_delta"]
        predicted = nominal(data, physical)[0]
        row.update(
            qualified_forward_input=data,
            command=command,
            direct_inverse=solved,
            prediction_kind="physical nominal only;direct inverse does not predict next measured state",
            response_residual=dict(innovation=data["x_response_residual"][0].copy(), completed_periods=index),
            nominal_predicted_motion=predicted[:6],
            nominal_predicted_linear_velocity=predicted[6:9] / 0.05,
            predicted_linear_velocity=predicted[6:9] / 0.05,
        )
        arm = dict(command=np.asarray(old["command"]))
        arguments = (arm, row["desired_source"], row["site_velocity_reference"])
        if index == 0:
            corruptions = [
                ("command",),
                ("direct_inverse", "physical_delta"),
                ("direct_inverse", "residual_command"),
                ("direct_inverse", "lower"),
                ("qualified_forward_input", "x_previous_command"),
                ("qualified_forward_input", "x_response_residual"),
                ("response_residual", "innovation"),
                ("nominal_predicted_motion",),
            ]
            for keys in corruptions:
                broken = copy.deepcopy(row)
                value = broken
                for key in keys[:-1]:
                    value = value[key]
                changed = np.asarray(value[keys[-1]], dtype=float).copy()
                changed.flat[0] += 0.02
                value[keys[-1]] = changed
                try:
                    copy.deepcopy(replay).verify(broken, *arguments)
                except AssertionError:
                    rejected.append("/".join(keys))
                else:
                    raise AssertionError(f"Corruption accepted: {keys}")
        replay.verify(row, *arguments)
        if index + 1 < len(rows):
            history.observe_completed(data, poses[index + 1], rows[index + 1]["measured_twist"], physical)
        previous = command.copy()
    assert replay.steps == 164 and len(rejected) == 8
    e.recheck()
    root.mkdir()
    publish(
        root / "independent_verified.json",
        dict(
            complete=True,
            rows=164,
            independent_command_and_history=True,
            corruptions_rejected=rejected,
            source_sha256=e.hashes,
            physics_steps=0,
            evidence_level="recorded states with reconstructed commands/history; no counterfactual dynamics",
            control_qualified=False,
        ),
    )
    print("213 independent direct command/history binding164rows;8 corruptions rejected;no physics")


if __name__ == "__main__":
    main()
