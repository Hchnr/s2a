"""Continuous candidate caches on recorded endpoints: interface proof, not physical rollout."""

import inspect
import json
from pathlib import Path

import numpy as np
from arm_joint_wrench_independent import nominal_prediction
from audit_wam_position import Evidence
from coupled_pose_preview_independent import verify_preview
from current_coupled_independent import response
from executed_residual_independent import ExecutedResidualReplay
from full_coupled_arm_log_replay import FullCoupledArmLogReplay, innovation
from full_coupled_pose_independent import generated_twist, objective
from scipy.spatial.transform import Rotation

from s2a.coupled_pose_preview_runtime import solve_preview
from s2a.coupled_response_history import CoupledResponseHistory
from s2a.current_coupled_response import build_response
from s2a.durable_stage_supervisor import publish
from s2a.executed_command_residual import ExecutedCommandResidualOSC
from s2a.osc import OSCInterface
from s2a.runtime_response_history import RuntimeResponseHistory


def wire(value):
    return json.loads(json.dumps(value, default=lambda x: x.tolist(), allow_nan=False))


def main():
    from robosuite.controllers.base_controller import Controller
    from robosuite.controllers.osc import OperationalSpaceController
    from robosuite.utils import control_utils, transform_utils

    root = Path("outputs/coupled_pose_preview_continuous_chain_20261026")
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "src/s2a/coupled_pose_preview_runtime.py",
        "src/s2a/coupled_pose_preview_executable.py",
        "src/s2a/coupled_pose_preview.py",
        "src/s2a/current_coupled_response.py",
        "src/s2a/strict_convex_qp.py",
        "src/s2a/full_coupled_pose_inverse.py",
        "src/s2a/coupled_response_history.py",
        "src/s2a/runtime_response_history.py",
        "src/s2a/executed_command_residual.py",
        "src/s2a/generated_reference_twist.py",
        "src/s2a/reference_linear_velocity.py",
        "scripts/coupled_pose_preview_independent.py",
        "scripts/current_coupled_independent.py",
        "scripts/full_coupled_arm_log_replay.py",
        "scripts/full_coupled_pose_independent.py",
        "scripts/executed_residual_independent.py",
    ):
        e.read(path)
    for obj in (OperationalSpaceController, Controller, transform_utils, control_utils):
        e.read(inspect.getfile(obj))
    admitted = e.json("outputs/coupled_pose_preview_native_commits_bounded_20261026/verified.json")
    assert admitted["complete"] and sum(x["periods"] for x in admitted["results"]) == 4907
    for path, digest in admitted["source_sha256"].items():
        e.check(path, digest)
    root.mkdir()
    results = []
    for case_index in range(18):
        directory = Path("outputs/current_coupled_control_20261026") / f"case{case_index:02d}"
        case = next(directory.glob("simulation/*/*"))
        rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
        ctl = e.json(case / "controller.json")
        contract = ctl["contract"]
        chunks = e.json(case / "chunks.json")
        config = e.json(directory / "execution_config.json")
        parameter_ref = config["arm_forward_parameter_ref"]
        e.check(parameter_ref["path"], parameter_ref["sha256"])
        parameters = e.json(parameter_ref["path"])
        manifest = e.json(case / "manifest.json")
        mapping = (
            np.linalg.inv(e.json(manifest["runtime_config"]["tool_contract"])["source_from_tcp"])
            @ ctl["source_from_site"]
        )
        reference = e.json(directory / "collection_result.json")["arrays_ref"]
        e.check(reference["path"], reference["sha256"])
        with np.load(reference["path"], allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in ("sim_states", "controller_site_pose", "controller_goal_ori")}
        xml = case / "model.xml"
        e.check(xml, manifest["initial_model_xml_sha256"])
        reader = FullCoupledArmLogReplay(parameters, contract, ctl["source_from_site"], xml, arrays)
        interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
        options = dict(compensate=True, inverse_weight=0.5, max_command_rate=5)
        source = ExecutedCommandResidualOSC(interface, contract["kp"], contract["kd"], 0.05, **options)
        source_check = ExecutedResidualReplay(interface, contract["kp"], contract["kd"], 0.05, **options)
        old_history, coupled = RuntimeResponseHistory(), CoupledResponseHistory()
        last_old, last_coupled = None, None
        previous = np.zeros(7)
        native = OperationalSpaceController.__new__(OperationalSpaceController)
        native.update = lambda: None
        native.impedance_mode = "fixed"
        native.use_delta = native.use_ori = True
        native.position_limits = native.orientation_limits = native.interpolator_pos = native.interpolator_ori = None
        native.action_scale = None
        for key in ("input_min", "input_max", "output_min", "output_max"):
            setattr(native, key, np.asarray(contract[key]))
        native.goal_ori = np.asarray(rows[0]["controller_goal_before"])
        native.goal_pos = np.asarray(rows[0]["controller_goal_position_before"])
        maximum = 0.0
        with (root / f"case{case_index:02d}.jsonl").open("x") as log:
            for index, row in enumerate(rows):
                pose, twist, mass, jacobian, *_ = reader.endpoint(index)
                packet = {k: np.asarray(v).copy() for k, v in row["qualified_forward_input"].items()}
                packet["x_previous_command"] = previous[None].copy()
                packet["x_previous_command_available"] = np.array([bool(index)])
                packet["x_previous_goal_ori"] = native.goal_ori[None].copy()
                packet["x_previous_goal_pos"] = native.goal_pos[None].copy()
                packet = old_history.attach(packet)
                np.testing.assert_allclose(
                    packet["x_response_residual"][0],
                    innovation(last_old, pose, twist, scaled_velocity=True),
                    atol=1e-8,
                    rtol=0,
                )
                bias = coupled.current(pose, twist)
                np.testing.assert_allclose(
                    bias, innovation(last_coupled, pose, twist, scaled_velocity=False), atol=1e-8, rtol=0
                )
                chunk = chunks[row["chunk_index"]]
                targets = np.asarray(chunk["effective_targets_tcp"]) @ mapping
                anchor = np.asarray(chunk["anchor_tcp"]) @ mapping
                cursor = row["site_velocity_reference"]["current_index"]
                solved = solve_preview(
                    mass, jacobian, pose, twist, targets, anchor, cursor, previous[:6], contract, innovation=bias
                )
                solved = wire(solved)
                certificate = verify_preview(
                    solved["preview"],
                    pose,
                    twist,
                    solved["references"],
                    solved["reference_twists"],
                    previous[:6],
                    mass,
                    jacobian,
                    contract,
                    innovation=bias,
                )
                for j, saved_twist in enumerate(solved["reference_twists"]):
                    np.testing.assert_allclose(
                        saved_twist, generated_twist(targets, cursor + j, anchor, 0.05)["twist"], atol=1e-10, rtol=0
                    )
                np.testing.assert_allclose(
                    solved["references"], targets[cursor : cursor + len(solved["references"])], atol=1e-12, rtol=0
                )
                command = np.r_[solved["command"], row["command"][-1]]
                checked = objective(
                    command[:6],
                    mass,
                    jacobian,
                    contract,
                    pose,
                    targets[cursor],
                    twist,
                    solved["reference_twists"][0],
                    bias,
                    native.goal_ori,
                    "reset",
                )
                for key in ("physical_delta", "nominal_response", "predicted_pose", "predicted_twist"):
                    np.testing.assert_allclose(solved[key], checked[key], atol=1e-8, rtol=0)
                native.ee_pos, native.ee_ori_mat = pose[:3, 3].copy(), pose[:3, :3].copy()
                native.set_goal(command[:6])
                delta = native.scale_action(command[:6])
                assert np.any(delta[3:] != 0)
                relative = native.goal_ori @ pose[:3, :3].T
                model = build_response(mass, jacobian, contract["kp"], contract["kd"], relative)
                expected = response(mass, jacobian, contract["kp"], contract["kd"], relative)
                nominal = (
                    expected["initial_velocity_response"] @ twist
                    + expected["goal_forcing_response"] @ np.r_[delta[:3], expected["rotation_forcing"]]
                )
                commit = coupled.commit_executed(pose, twist, model, command[:3], contract)
                np.testing.assert_allclose(commit["nominal_response"], nominal, atol=1e-8, rtol=0)
                measured = np.asarray(row["measured_source"])
                args = (measured, np.asarray(row["desired_source"]), twist)
                kw = dict(desired_linear_velocity=np.asarray(row["linear_velocity_reference"]["velocity"]))
                source.convert(*args, **kw)
                source_check.convert(*args, **kw)
                actual_source = source.commit_executed(measured, twist, command[:6])
                checked_source = source_check.commit_executed(measured, twist, command[:6])
                for key in checked_source:
                    np.testing.assert_allclose(actual_source[key], checked_source[key], atol=1e-9, rtol=0)
                packet["action_command"] = command[None].copy()
                packet["action_goal_delta"] = np.r_[delta[:3], Rotation.from_matrix(relative).as_rotvec()][None]
                old_nominal = nominal_prediction(packet, parameters)[0]
                next_pose, next_twist, *_ = reader.endpoint(index + 1)
                after = coupled.observe_completed(next_pose, next_twist)
                old_history.observe_completed(packet, next_pose, next_twist, parameters)
                np.testing.assert_allclose(
                    after, innovation((pose, nominal), next_pose, next_twist, scaled_velocity=False), atol=1e-8, rtol=0
                )
                np.testing.assert_allclose(
                    old_history.innovation,
                    innovation((pose, old_nominal), next_pose, next_twist, scaled_velocity=True),
                    atol=1e-8,
                    rtol=0,
                )
                assert old_history.completed_periods == coupled.completed_periods == index + 1
                record = wire(
                    dict(
                        step=index,
                        command=command,
                        solution=solved,
                        certificate=certificate,
                        coupled_commit=commit,
                        source_commit=actual_source,
                        coupled_after=after,
                        old_after=old_history.innovation,
                    )
                )
                np.testing.assert_array_equal(record["command"][:6], record["solution"]["preview"]["command"])
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                maximum = max(maximum, certificate["max_independent_stationarity"])
                previous = command.copy()
                last_old = (pose.copy(), old_nominal.copy())
                last_coupled = (pose.copy(), nominal.copy())
        e.read(root / f"case{case_index:02d}.jsonl")
        results.append(dict(case=case_index, periods=len(rows), maximum_kkt=maximum, source_commits=len(rows)))
        print(results[-1], flush=True)
    e.recheck()
    publish(
        root / "verified.json",
        dict(
            complete=True,
            seed=42,
            results=results,
            source_sha256=e.hashes,
            scope="continuous reconstructed candidate caches on recorded raw endpoints; not counterfactual physics",
            new_physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
