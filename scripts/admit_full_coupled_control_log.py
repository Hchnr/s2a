"""234 JSON control-log admission on recorded states; no counterfactual physics."""

import copy
import ctypes
import json
import os
import time

import numpy as np
from audit_wam_position import Evidence
from executed_residual_independent import ExecutedResidualReplay
from full_coupled_arm_log_replay import FullCoupledArmLogReplay
from full_coupled_pose_independent import generated_twist
from run_matched_oracle_case import CONTROL, CONTROL_SHA
from scipy.spatial.transform import Rotation

from s2a.archived_forward_state import verify_forward_state
from s2a.artifacts import REPO, sha256
from s2a.coupled_response_history import CoupledResponseHistory
from s2a.current_coupled_response import build_response
from s2a.current_coupled_state import rotation_goal_from_command
from s2a.durable_stage_supervisor import publish
from s2a.executed_command_residual import ExecutedCommandResidualOSC
from s2a.full_coupled_pose_inverse import FullCoupledPoseModel, solve_pose
from s2a.generated_reference_twist import reference_world_twist
from s2a.joint_wrench_condition_forward import nominal
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.runtime_response_history import RuntimeResponseHistory


def serialized(value):
    return json.loads(json.dumps(value, default=lambda item: item.tolist(), allow_nan=False))


def compare(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        if isinstance(expected[key], dict):
            compare(actual[key], expected[key])
        elif expected[key] is None or isinstance(expected[key], str):
            assert actual[key] == expected[key]
        else:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-9, rtol=0)


def main():
    strict_native_environment()
    assert sha256(CONTROL) == CONTROL_SHA
    guard = json.loads(CONTROL.read_text())["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    root = REPO / "outputs/full_coupled_control_log_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "scripts/full_coupled_arm_log_replay.py",
        "scripts/full_coupled_pose_independent.py",
        "scripts/current_coupled_independent.py",
        "scripts/executed_residual_independent.py",
        "scripts/arm_joint_wrench_independent.py",
        "src/s2a/full_coupled_control_simulation.py",
        "src/s2a/full_coupled_pose_inverse.py",
        "src/s2a/generated_reference_twist.py",
        "src/s2a/current_coupled_response.py",
        "src/s2a/current_coupled_state.py",
        "src/s2a/coupled_response_history.py",
        "src/s2a/archived_forward_state.py",
        "src/s2a/executed_command_residual.py",
        "src/s2a/runtime_response_history.py",
        "src/s2a/joint_wrench_condition_forward.py",
        "plans/v0.0.1/242_full_pose_control_log_binding_20261026.md",
    ):
        e.read(path)
    for path, required in (
        ("outputs/full_coupled_chain_supervision_20261026/execution_result.json", "actual_owned_wait_recorded"),
        (
            "outputs/full_coupled_recorded_chain_20261026/independent_verified.json",
            "independent_full_pose_inverse_and_native_history",
        ),
        ("outputs/current_coupled_reader_admission_20261026/verified.json", "current_dynamics_reader_verified"),
    ):
        proof = e.json(path)
        assert proof["complete"] and proof[required]
        for source, signature in proof.get("source_sha256", {}).items():
            e.check(source, signature)
    directory = REPO / "outputs/current_coupled_control_20261026/case00"
    case = next((directory / "simulation").glob("*/*"))
    rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
    assert len(rows) == 164
    controller = e.json(case / "controller.json")
    contract = controller["contract"]
    config = e.json(directory / "execution_config.json")
    chunks = e.json(case / "chunks.json")
    template = e.json(config["policy_template"])
    h = np.asarray(e.json(template["tool_contract"])["source_from_tcp"])
    tcp_from_site = np.linalg.inv(h) @ np.asarray(controller["source_from_site"])
    control_rotation = np.asarray(controller["source_from_control_rotation"])
    parameter_ref = config["arm_forward_parameter_ref"]
    e.check(parameter_ref["path"], parameter_ref["sha256"])
    parameters = e.json(parameter_ref["path"])
    ref = e.json(directory / "collection_result.json")["arrays_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in ("sim_states", "controller_site_pose", "controller_goal_ori")}
    manifest = e.json(case / "manifest.json")
    xml = case / "model.xml"
    e.check(xml, manifest["initial_model_xml_sha256"])
    ctypes.CDLL(manifest["runtime_config"]["osmesa_library"], mode=ctypes.RTLD_GLOBAL)
    root.mkdir()
    os.environ.update(NUMBA_CACHE_DIR=str(root / "numba_cache"), MPLCONFIGDIR=str(root / "matplotlib"))
    import mujoco

    # Producer restores its own raw model; the checker has a separate model/data.
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
    assert sid >= 0
    qids, dofs, segments = [], [], []
    for joint in range(1, 8):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{joint}")
        assert jid >= 0
        qids.append(model.jnt_qposadr[jid])
        dofs.append(model.jnt_dofadr[jid])
    for joint in range(model.njnt):
        kind = int(model.jnt_type[joint])
        if kind in (0, 1):
            segments.append(
                dict(
                    joint=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
                    qpos_start=int(model.jnt_qposadr[joint]) + (3 if kind == 0 else 0),
                )
            )

    def endpoint(index):
        state = arrays["sim_states"][index]
        data.time, data.qpos[:], data.qvel[:] = state[0], state[1 : 1 + model.nq], state[1 + model.nq :]
        mujoco.mj_forward(model, data)
        verify_forward_state(state, np.r_[data.time, data.qpos, data.qvel], nq=model.nq, quaternion_segments=segments)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = data.site_xmat[sid].reshape(3, 3), data.site_xpos[sid]
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, sid)
        full_j = np.vstack([jp, jr])
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        return pose, full_j @ data.qvel, mass[np.ix_(dofs, dofs)], full_j[:, dofs]

    interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    options = dict(compensate=True, inverse_weight=0.5, max_command_rate=5)
    converter = ExecutedCommandResidualOSC(interface, contract["kp"], contract["kd"], 0.05, **options)
    source_replay = ExecutedResidualReplay(interface, contract["kp"], contract["kd"], 0.05, **options)

    def new_replay():
        return FullCoupledArmLogReplay(parameters, contract, controller["source_from_site"], xml, arrays)

    replay, old_history, coupled_history = new_replay(), RuntimeResponseHistory(), CoupledResponseHistory()
    previous, rejected, completed = np.zeros(7), [], 0
    previous_goal_position = np.asarray(rows[0]["controller_goal_position_before"])
    try:
        with (root / "reconstructed_steps.jsonl").open("x") as log:
            for index, old in enumerate(rows):
                row = copy.deepcopy(old)
                pose, twist, mass, jacobian = endpoint(index)
                packet = {key: np.asarray(value).copy() for key, value in row["qualified_forward_input"].items()}
                packet["x_previous_command"] = previous[None].copy()
                packet["x_previous_command_available"] = np.array([bool(index)])
                packet["x_previous_goal_ori"][0] = arrays["controller_goal_ori"][index]
                packet = old_history.attach(packet)
                measured, desired = (np.asarray(row[key]) for key in ("measured_source", "desired_source"))
                args = (measured, desired, twist)
                kw = dict(desired_linear_velocity=np.asarray(row["linear_velocity_reference"]["velocity"]))
                converted, arm = converter.convert(*args, **kw), source_replay.convert(*args, **kw)
                np.testing.assert_allclose(converted["command"], arm["command"], atol=1e-9, rtol=0)
                compare(converted["response_residual"], arm["response_residual"])
                previous_goal = arrays["controller_goal_ori"][index].copy()
                chunk = chunks[row["chunk_index"]]
                targets = np.asarray(chunk["effective_targets_tcp"]) @ tcp_from_site
                anchor = np.asarray(chunk["anchor_tcp"]) @ tcp_from_site
                site_twist = reference_world_twist(targets, index - chunk["step"], anchor, 0.05)
                before = coupled_history.current(pose, twist)
                start = time.monotonic()
                proposal_model = FullCoupledPoseModel(mass, jacobian, contract["kp"], contract["kd"])
                solution = solve_pose(
                    proposal_model,
                    pose,
                    np.asarray(row["desired_site"]),
                    twist,
                    site_twist["twist"],
                    previous[:6],
                    previous_goal,
                    contract,
                    innovation=before,
                )
                solve_seconds = time.monotonic() - start
                command = np.r_[solution["command"], old["command"][-1]]
                goal, relative = rotation_goal_from_command(command[:6], interface, pose[:3, :3], previous_goal)
                arrays["controller_goal_ori"][index + 1] = goal
                dynamics = dict(
                    mass=mass,
                    jacobian=jacobian,
                    robot_joint_position=data.qpos[qids].copy(),
                    robot_joint_velocity=data.qvel[dofs].copy(),
                    kp=np.asarray(contract["kp"]),
                    kd=np.asarray(contract["kd"]),
                    actual_rotation_goal=goal,
                    relative_rotation_goal=relative,
                    uncoupling=True,
                    use_delta=True,
                    impedance_mode="fixed",
                    control_freq=20,
                    model_timestep=0.002,
                )
                response = build_response(mass, jacobian, contract["kp"], contract["kd"], relative)
                commit = coupled_history.commit_executed(pose, twist, response, command[:3], contract)
                continuous_goal = (
                    previous_goal
                    if solution["goal_mode"] == "latch"
                    else Rotation.from_rotvec(solution["physical_delta"][3:]).as_matrix() @ pose[:3, :3]
                )
                difference = np.abs(solution["nominal_response"] - commit["nominal_response"])
                comparison = dict(
                    continuous_goal=continuous_goal,
                    native_goal_max_difference=float(np.max(np.abs(goal - continuous_goal))),
                    native_response_block_max_difference=[
                        float(np.max(difference[part]))
                        for part in (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12))
                    ],
                    response_block_units=["m", "rad", "m/s", "rad/s"],
                )
                source_commit = converter.commit_executed(measured, twist, command[:6])
                compare(source_commit, source_replay.commit_executed(measured, twist, command[:6]))
                packet["action_command"][0] = command
                packet["action_goal_delta"][0, :3] = solution["physical_delta"][:3]
                packet["action_goal_delta"][0, 3:] = Rotation.from_matrix(relative).as_rotvec()
                old_prediction = nominal(packet, parameters)[0]
                next_pose, next_twist, *_ = endpoint(index + 1)
                after = coupled_history.observe_completed(next_pose, next_twist)
                row.update(
                    qualified_forward_input=packet,
                    command=command,
                    saturated=bool(
                        np.any(np.isclose(command[:6], contract["input_min"], atol=1e-10))
                        or np.any(np.isclose(command[:6], contract["input_max"], atol=1e-10))
                    ),
                    coupled_dynamics=dynamics,
                    coupled_response_model=response,
                    full_pose_inverse=solution,
                    full_pose_solver_seconds=solve_seconds,
                    full_pose_goal_comparison=comparison,
                    provisional_source_command=converted["command"].copy(),
                    site_twist_reference=site_twist,
                    controller_goal_before=previous_goal,
                    controller_goal_after=goal,
                    controller_goal_source_rotation_before=previous_goal @ control_rotation.T,
                    controller_goal_position_before=previous_goal_position,
                    controller_goal_position_after=measured[:3, 3] + solution["physical_delta"][:3],
                    coupled_response_history=dict(
                        innovation_before=before,
                        completed_periods_before=index,
                        commit=commit,
                        innovation_after=after,
                        completed_periods_after=index + 1,
                    ),
                    physical_response_residual=converted["response_residual"],
                    physical_response_commit=source_commit,
                    response_residual=dict(innovation=packet["x_response_residual"][0], completed_periods=index),
                    nominal_predicted_motion=old_prediction[:6],
                    nominal_predicted_linear_velocity=old_prediction[6:9] / 0.05,
                    predicted_linear_velocity=solution["predicted_twist"][:3],
                    prediction_kind="239 continuous full-pose proposal;completed innovation;native nominal separately",
                    nominal_model_kind="150 original uncoupled nominal;coupled nominal recorded separately",
                    rate_limited=bool(np.any(np.isclose(np.abs(command[:6] - previous[:6]), 0.25, atol=1e-10))),
                )
                row = serialized(row)
                arguments = (
                    arm,
                    row["desired_source"],
                    row["site_velocity_reference"],
                    generated_twist(targets, index - chunk["step"], anchor, 0.05),
                )
                if index == 0:
                    paths = [
                        ("command",),
                        ("qualified_forward_input", "x_joint_position"),
                        ("qualified_forward_input", "x_previous_command"),
                        ("qualified_forward_input", "x_response_residual"),
                        ("coupled_dynamics", "mass"),
                        ("coupled_dynamics", "jacobian"),
                        ("coupled_response_model", "initial_velocity_response"),
                        ("full_pose_inverse", "physical_delta"),
                        ("full_pose_inverse", "predicted_pose"),
                        ("coupled_response_history", "innovation_before"),
                        ("coupled_response_history", "innovation_after"),
                        ("coupled_response_history", "commit", "nominal_response"),
                        ("coupled_response_history", "commit", "executed_translation"),
                        ("desired_site",),
                        ("controller_goal_after",),
                        ("nominal_predicted_motion",),
                        ("provisional_source_command",),
                        ("site_twist_reference", "twist"),
                        ("full_pose_goal_comparison", "native_goal_max_difference"),
                        ("full_pose_inverse", "mode_candidates", 0, "gradient"),
                    ]
                    for path in paths:
                        broken = copy.deepcopy(row)
                        obj = broken
                        for key in path[:-1]:
                            obj = obj[key]
                        changed = np.asarray(obj[path[-1]], dtype=float).copy()
                        changed.flat[0] += 0.02
                        obj[path[-1]] = changed.tolist()
                        try:
                            new_replay().verify(broken, *arguments)
                        except AssertionError:
                            rejected.append("/".join(map(str, path)))
                        else:
                            raise AssertionError(f"Corrupt coupled log accepted: {path}")
                replay.verify(row, *arguments)
                compare(row["physical_response_residual"], arm["response_residual"])
                compare(row["physical_response_commit"], source_commit)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                old_history.observe_completed(packet, next_pose, next_twist, parameters)
                previous_goal_position = np.asarray(row["controller_goal_position_after"])
                previous, completed = command.copy(), index + 1
        assert replay.steps == completed == 164 and len(rejected) == 20
        with (root / "reconstructed_controller_goals.npy").open("xb") as stream:
            np.save(stream, arrays["controller_goal_ori"], allow_pickle=False)
        e.read(root / "reconstructed_controller_goals.npy")
        e.read(root / "reconstructed_steps.jsonl")
        e.recheck()
        publish(
            root / "independent_verified.json",
            dict(
                complete=True,
                rows=completed,
                json_roundtrip_rows=completed,
                source_commit_cycles=completed,
                independent_full_pose_control_log=True,
                reconstructed_controller_goals=True,
                corruptions_rejected=rejected,
                max_stationarity_residual=replay.maximum_kkt,
                normalization_checks=replay.normalization_checks,
                source_sha256=e.hashes,
                physics_steps=0,
                new_fits=0,
                model_calls=0,
                seed=42,
                evidence_level="recorded robot endpoints/reconstructed controller goal cache, commands and dual history;no counterfactual physical trajectory",
                control_qualified=False,
            ),
        )
    except Exception as error:
        publish(
            root / "failure.json",
            dict(
                complete=False,
                checked_rows=completed,
                error_type=type(error).__name__,
                error=str(error),
                source_sha256=e.hashes,
                physics_steps=0,
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
