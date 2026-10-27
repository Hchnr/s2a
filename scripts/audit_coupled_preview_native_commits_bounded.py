"""Isolated native decoding and history commits for recorded-state preview proposals."""

import inspect
import json
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from current_coupled_independent import response as independent_response
from executed_residual_independent import ExecutedResidualReplay
from full_coupled_arm_log_replay import innovation
from scipy.spatial.transform import Rotation

from s2a.coupled_response_history import CoupledResponseHistory
from s2a.current_coupled_response import build_response
from s2a.durable_stage_supervisor import publish
from s2a.executed_command_residual import ExecutedCommandResidualOSC
from s2a.osc import OSCInterface


def main():
    from robosuite.controllers.osc import OperationalSpaceController
    from robosuite.utils import control_utils, transform_utils

    output = Path("outputs/coupled_pose_preview_native_commits_bounded_20261026/verified.json")
    assert not output.exists()
    e = Evidence()
    for path in (
        __file__,
        "src/s2a/coupled_response_history.py",
        "src/s2a/current_coupled_response.py",
        "scripts/current_coupled_independent.py",
        "src/s2a/executed_command_residual.py",
        "src/s2a/causal_arm_residual.py",
        "scripts/executed_residual_independent.py",
        "scripts/causal_arm_independent_replay.py",
        "scripts/full_coupled_arm_log_replay.py",
        "src/s2a/osc.py",
        "src/s2a/geometry.py",
    ):
        e.read(path)
    for obj in (OperationalSpaceController, control_utils, transform_utils):
        e.read(inspect.getfile(obj))
    from robosuite.controllers.base_controller import Controller

    e.read(inspect.getfile(Controller))
    proposal = e.json("outputs/coupled_pose_preview_records_bounded_20261026/verified.json")
    binding = e.json("outputs/coupled_pose_preview_records_bounded_20261026/input_binding_verified.json")
    assert proposal["complete"] and binding["complete"] and len(binding["bindings"]) == 90
    for path, digest in proposal["source_sha256"].items():
        e.check(path, digest)
    results = []
    for case_proposal in proposal["cases"]:
        index = case_proposal["case"]
        root = Path("outputs/current_coupled_control_20261026") / f"case{index:02d}"
        case = next(root.glob("simulation/*/*"))
        controller_record = e.json(case / "controller.json")
        contract = controller_record["contract"]
        interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
        rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
        controller = OperationalSpaceController.__new__(OperationalSpaceController)
        controller.update = lambda: None
        controller.impedance_mode = "fixed"
        controller.use_delta = controller.use_ori = True
        controller.position_limits = controller.orientation_limits = None
        controller.interpolator_pos = controller.interpolator_ori = None
        controller.action_scale = None
        for key in ("input_min", "input_max", "output_min", "output_max"):
            setattr(controller, key, np.asarray(contract[key]))
        maximum_response = 0.0
        source_commits = 0
        for row, saved in zip(rows, case_proposal["checks"], strict=True):
            assert row["step"] == saved["step"]
            command = np.asarray(json.loads(json.dumps(saved["command"], allow_nan=False)))
            np.testing.assert_array_equal(command, saved["command"])
            source_pose = np.asarray(row["measured_source"])
            pose = source_pose @ np.asarray(controller_record["source_from_site"])
            twist = np.asarray(row["measured_twist"])
            controller.ee_pos, controller.ee_ori_mat = pose[:3, 3].copy(), pose[:3, :3].copy()
            controller.goal_ori = np.asarray(row["controller_goal_before"]).copy()
            before = controller.goal_ori.copy()
            controller.set_goal(command)
            delta = controller.scale_action(command)
            expected_goal = (
                before
                if np.all(delta[3:] == 0)
                else transform_utils.quat2mat(transform_utils.axisangle2quat(delta[3:])) @ pose[:3, :3]
            )
            np.testing.assert_array_equal(controller.goal_ori, expected_goal)
            relative = controller.goal_ori @ pose[:3, :3].T
            dynamics = row["coupled_response_model"]
            model = build_response(dynamics["mass"], dynamics["jacobian"], contract["kp"], contract["kd"], relative)
            independent = independent_response(
                dynamics["mass"], dynamics["jacobian"], contract["kp"], contract["kd"], relative
            )
            expected = (
                independent["initial_velocity_response"] @ twist
                + independent["goal_forcing_response"] @ np.r_[delta[:3], independent["rotation_forcing"]]
            )
            # Fresh history per recorded sample: no claim of candidate-generated trajectory.
            history = CoupledResponseHistory()
            commit = history.commit_executed(pose, twist, model, command[:3], contract)
            difference = float(np.max(np.abs(commit["nominal_response"] - expected)))
            maximum_response = max(maximum_response, difference)
            np.testing.assert_allclose(commit["nominal_response"], expected, atol=1e-8, rtol=0)
            # Synthetic endpoint checks history units/composition, not simulator accuracy.
            after = pose.copy()
            after[:3, 3] += expected[:3]
            after[:3, :3] = Rotation.from_rotvec(expected[3:6]).as_matrix() @ pose[:3, :3]
            after_twist = expected[6:]
            actual = history.observe_completed(after, after_twist)
            independent_bias = innovation((pose, expected), after, after_twist, scaled_velocity=False)
            np.testing.assert_allclose(actual, independent_bias, atol=1e-8, rtol=0)
            options = dict(compensate=True, inverse_weight=0.5, max_command_rate=5)
            source = ExecutedCommandResidualOSC(interface, contract["kp"], contract["kd"], 0.05, **options)
            replay = ExecutedResidualReplay(interface, contract["kp"], contract["kd"], 0.05, **options)
            arguments = (source_pose, np.asarray(row["desired_source"]), twist)
            kw = dict(desired_linear_velocity=np.asarray(row["linear_velocity_reference"]["velocity"]))
            source.convert(*arguments, **kw)
            replay.convert(*arguments, **kw)
            committed = source.commit_executed(source_pose, twist, command)
            checked = replay.commit_executed(source_pose, twist, command)
            for key in checked:
                np.testing.assert_allclose(committed[key], checked[key], atol=1e-10, rtol=0)
            np.testing.assert_array_equal(source.previous_command, command)
            try:
                source.commit_executed(source_pose, twist, command)
            except ValueError:
                pass
            else:
                raise AssertionError("duplicate SOURCE commit accepted")
            source_commits += 1
        results.append(
            dict(
                case=index,
                periods=len(rows),
                source_commits=source_commits,
                maximum_independent_native_response_difference=maximum_response,
            )
        )
        print(results[-1], flush=True)
    e.recheck()
    publish(
        output,
        dict(
            complete=True,
            seed=42,
            results=results,
            source_sha256=e.hashes,
            scope="isolated recorded-state native goal and SOURCE/coupled commits; synthetic endpoints only",
            continuous_dual_history_qualified=False,
            new_physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
