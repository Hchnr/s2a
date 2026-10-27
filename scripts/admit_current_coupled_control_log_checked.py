"""234 JSON control-log admission on recorded states; no counterfactual physics."""

import copy
import ctypes
import json
import os

import numpy as np
from audit_wam_position import Evidence
from current_coupled_arm_log_replay import CurrentCoupledArmLogReplay
from executed_residual_independent import ExecutedResidualReplay
from run_matched_oracle_case import CONTROL, CONTROL_SHA

from s2a.archived_forward_state import verify_forward_state
from s2a.artifacts import REPO, sha256
from s2a.coupled_response_history import CoupledResponseHistory
from s2a.current_coupled_response import build_response, solve_translation
from s2a.current_coupled_state import rotation_goal_from_command
from s2a.durable_stage_supervisor import publish
from s2a.executed_command_residual import ExecutedCommandResidualOSC
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
    root = REPO / "outputs/current_coupled_control_log_checked_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "scripts/admit_current_coupled_control_log.py",
        "outputs/current_coupled_control_log_20261026/failure.json",
        "outputs/current_coupled_log_supervision_20261026/verification/exit.json",
        "outputs/current_coupled_log_supervision_20261026/verification/release.json",
        "plans/v0.0.1/235_coupled_log_comparison_repair_20261026.md",
        "scripts/current_coupled_arm_log_replay.py",
        "scripts/current_coupled_independent.py",
        "scripts/executed_residual_independent.py",
        "scripts/arm_joint_wrench_independent.py",
        "src/s2a/current_coupled_control_simulation.py",
        "src/s2a/current_coupled_response.py",
        "src/s2a/current_coupled_state.py",
        "src/s2a/coupled_response_history.py",
        "src/s2a/archived_forward_state.py",
        "src/s2a/executed_command_residual.py",
        "src/s2a/runtime_response_history.py",
        "src/s2a/joint_wrench_condition_forward.py",
        "plans/v0.0.1/234_coupled_control_log_binding_20261026.md",
    ):
        e.read(path)
    for path, required in (
        (
            "outputs/coupled_recorded_chain_supervision_checked_20261026/execution_result.json",
            "actual_owned_wait_recorded",
        ),
        (
            "outputs/current_coupled_recorded_chain_checked_20261026/independent_verified.json",
            "independent_coupled_inverse_and_history",
        ),
        ("outputs/current_coupled_reader_admission_20261026/verified.json", "current_dynamics_reader_verified"),
    ):
        proof = e.json(path)
        assert proof["complete"] and proof[required]
        for source, signature in proof.get("source_sha256", {}).items():
            e.check(source, signature)
    directory = REPO / "outputs/preview_control_20261026/case00"
    case = next((directory / "simulation").glob("*/*"))
    rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
    assert len(rows) == 164
    controller = e.json(case / "controller.json")
    contract = controller["contract"]
    config = e.json(directory / "execution_config.json")
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
        return CurrentCoupledArmLogReplay(parameters, contract, controller["source_from_site"], xml, arrays)

    replay, old_history, coupled_history = new_replay(), RuntimeResponseHistory(), CoupledResponseHistory()
    previous, rejected, completed = np.zeros(7), [], 0
    try:
        with (root / "reconstructed_steps.jsonl").open("x") as log:
            for index, old in enumerate(rows):
                row = copy.deepcopy(old)
                pose, twist, mass, jacobian = endpoint(index)
                packet = {key: np.asarray(value).copy() for key, value in row["qualified_forward_input"].items()}
                packet["x_previous_command"] = previous[None].copy()
                packet["x_previous_command_available"] = np.array([bool(index)])
                packet = old_history.attach(packet)
                measured, desired = (np.asarray(row[key]) for key in ("measured_source", "desired_source"))
                args = (measured, desired, twist)
                kw = dict(desired_linear_velocity=np.asarray(row["linear_velocity_reference"]["velocity"]))
                converted, arm = converter.convert(*args, **kw), source_replay.convert(*args, **kw)
                np.testing.assert_allclose(converted["command"], arm["command"], atol=1e-9, rtol=0)
                compare(converted["response_residual"], arm["response_residual"])
                command = np.r_[converted["command"], old["command"][-1]]
                goal, relative = rotation_goal_from_command(
                    command[:6], interface, pose[:3, :3], arrays["controller_goal_ori"][index]
                )
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
                before = coupled_history.current(pose, twist)
                solution = solve_translation(
                    pose[:3, 3],
                    twist,
                    np.asarray(row["desired_site"])[:3, 3],
                    row["site_velocity_reference"]["velocity"],
                    previous[:3],
                    contract,
                    response,
                    innovation=np.vstack([before[:3], before[6:9] / 0.05]),
                )
                command[:3] = solution["command"]
                commit = coupled_history.commit_executed(pose, twist, response, command[:3], contract)
                source_commit = converter.commit_executed(measured, twist, command[:6])
                compare(source_commit, source_replay.commit_executed(measured, twist, command[:6]))
                packet["action_command"][0] = command
                packet["action_goal_delta"][0, :3] = solution["physical_delta"]
                # Rotation is unchanged; bind its current native goal, too.
                from scipy.spatial.transform import Rotation

                packet["action_goal_delta"][0, 3:] = Rotation.from_matrix(relative).as_rotvec()
                old_prediction = nominal(packet, parameters)[0]
                next_pose, next_twist, *_ = endpoint(index + 1)
                after = coupled_history.observe_completed(next_pose, next_twist)
                row.update(
                    qualified_forward_input=packet,
                    command=command,
                    coupled_dynamics=dynamics,
                    coupled_response_model=response,
                    coupled_inverse=solution,
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
                    predicted_linear_velocity=solution["predicted_response"][6:9],
                    prediction_kind="228 current coupled local response;completed innovation;not measured future",
                    nominal_model_kind="150 original uncoupled nominal;coupled nominal recorded separately",
                    rate_limited=bool(np.any(np.isclose(np.abs(command[:3] - previous[:3]), 0.25, atol=1e-10))),
                )
                row = serialized(row)
                arguments = (arm, row["desired_source"], row["site_velocity_reference"])
                if index == 0:
                    paths = [
                        ("command",),
                        ("qualified_forward_input", "x_joint_position"),
                        ("qualified_forward_input", "x_previous_command"),
                        ("qualified_forward_input", "x_response_residual"),
                        ("coupled_dynamics", "mass"),
                        ("coupled_dynamics", "jacobian"),
                        ("coupled_response_model", "initial_velocity_response"),
                        ("coupled_inverse", "physical_delta"),
                        ("coupled_inverse", "predicted_response"),
                        ("coupled_response_history", "innovation_before"),
                        ("coupled_response_history", "innovation_after"),
                        ("coupled_response_history", "commit", "nominal_response"),
                        ("coupled_response_history", "commit", "executed_translation"),
                        ("desired_site",),
                        ("controller_goal_after",),
                        ("nominal_predicted_motion",),
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
                            rejected.append("/".join(path))
                        else:
                            raise AssertionError(f"Corrupt coupled log accepted: {path}")
                replay.verify(row, *arguments)
                for key, value in arm["response_residual"].items():
                    np.testing.assert_allclose(row["physical_response_residual"][key], value, atol=1e-9, rtol=0)
                compare(row["physical_response_commit"], source_commit)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                old_history.observe_completed(packet, next_pose, next_twist, parameters)
                previous, completed = command.copy(), index + 1
        assert replay.steps == completed == 164 and len(rejected) == 16
        e.read(root / "reconstructed_steps.jsonl")
        e.recheck()
        publish(
            root / "independent_verified.json",
            dict(
                complete=True,
                rows=completed,
                json_roundtrip_rows=completed,
                source_commit_cycles=completed,
                independent_full_control_log=True,
                corruptions_rejected=rejected,
                max_stationarity_residual=replay.maximum_kkt,
                normalization_checks=replay.normalization_checks,
                source_sha256=e.hashes,
                physics_steps=0,
                new_fits=0,
                model_calls=0,
                seed=42,
                evidence_level="recorded states/reconstructed commands and dual history;no counterfactual physical trajectory",
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
