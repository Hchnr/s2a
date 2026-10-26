"""239 full-pose proposal/native-goal/history admission on164 recorded endpoints."""

import copy
import ctypes
import json
import os

import numpy as np
from audit_wam_position import Evidence
from current_coupled_independent import response as independent_response
from full_coupled_pose_independent import generated_twist, verify_pose_solution
from run_matched_oracle_case import CONTROL, CONTROL_SHA
from scipy.spatial.transform import Rotation

from s2a.archived_forward_state import verify_forward_state
from s2a.artifacts import REPO, sha256
from s2a.coupled_response_history import CoupledResponseHistory
from s2a.current_coupled_response import build_response
from s2a.current_coupled_state import rotation_goal_from_command
from s2a.durable_stage_supervisor import publish
from s2a.full_coupled_pose_inverse import FullCoupledPoseModel, solve_pose
from s2a.generated_reference_twist import reference_world_twist
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface


def serialized(value):
    return json.loads(json.dumps(value, default=lambda item: item.tolist(), allow_nan=False))


def main():
    strict_native_environment()
    assert sha256(CONTROL) == CONTROL_SHA
    guard = json.loads(CONTROL.read_text())["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    root = REPO / "outputs/full_coupled_recorded_chain_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "src/s2a/full_coupled_pose_inverse.py",
        "src/s2a/generated_reference_twist.py",
        "scripts/full_coupled_pose_independent.py",
        "scripts/current_coupled_independent.py",
        "src/s2a/current_coupled_response.py",
        "src/s2a/current_coupled_state.py",
        "src/s2a/coupled_response_history.py",
        "src/s2a/archived_forward_state.py",
        "tests/test_full_coupled_pose_inverse.py",
        "tests/test_generated_reference_twist.py",
        "plans/v0.0.1/239_nonlinear_coupled_pose_inverse_20261026.md",
    ):
        e.read(path)
    for path in (
        "outputs/current_coupled_control_20261026/closure_verified.json",
        "outputs/current_coupled_rotation_tracking_20261026/verified.json",
        "outputs/current_coupled_rotation_supervision_20261026/execution_result.json",
    ):
        proof = e.json(path)
        assert proof["complete"]
        for source, signature in proof.get("source_sha256", {}).items():
            e.check(source, signature)
    directory = REPO / "outputs/current_coupled_control_20261026/case00"
    case = next(directory.glob("simulation/*/*"))
    rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
    assert len(rows) == 164
    controller, chunks = e.json(case / "controller.json"), e.json(case / "chunks.json")
    config = e.json(directory / "execution_config.json")
    template = e.json(config["policy_template"])
    h = np.asarray(e.json(template["tool_contract"])["source_from_tcp"])
    tcp_from_site = np.linalg.inv(h) @ np.asarray(controller["source_from_site"])
    contract = controller["contract"]
    interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    ref = e.json(directory / "collection_result.json")["arrays_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as archive:
        states, poses = archive["sim_states"].copy(), archive["controller_site_pose"].copy()
    manifest = e.json(case / "manifest.json")
    ctypes.CDLL(manifest["runtime_config"]["osmesa_library"], mode=ctypes.RTLD_GLOBAL)
    import mujoco

    xml = case / "model.xml"
    e.check(xml, manifest["initial_model_xml_sha256"])
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
    assert sid >= 0
    dofs = []
    for joint in range(1, 8):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{joint}")
        assert jid >= 0
        dofs.append(model.jnt_dofadr[jid])
    segments = []
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
        state = states[index]
        data.time = state[0]
        data.qpos[:], data.qvel[:] = state[1 : 1 + model.nq], state[1 + model.nq :]
        mujoco.mj_forward(model, data)
        verify_forward_state(state, np.r_[data.time, data.qpos, data.qvel], nq=model.nq, quaternion_segments=segments)
        pose = np.eye(4)
        pose[:3, 3], pose[:3, :3] = data.site_xpos[sid], data.site_xmat[sid].reshape(3, 3)
        np.testing.assert_allclose(pose, poses[index], atol=1e-9, rtol=0)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, sid)
        jacobian = np.vstack([jp, jr])
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        return pose, jacobian @ data.qvel, mass[np.ix_(dofs, dofs)].copy(), jacobian[:, dofs].copy()

    root.mkdir()
    os.environ.update(NUMBA_CACHE_DIR=str(root / "numba_cache"), MPLCONFIGDIR=str(root / "matplotlib"))
    history, previous, previous_goal = (
        CoupledResponseHistory(),
        np.zeros(6),
        np.asarray(rows[0]["controller_goal_before"]),
    )
    independent_last = None
    rejected, results, completed = [], [], 0
    try:
        with (root / "reconstructed_steps.jsonl").open("x") as log:
            for index, original in enumerate(rows):
                pose, twist, mass, jacobian = endpoint(index)
                chunk = chunks[original["chunk_index"]]
                targets, anchor = (
                    np.asarray(chunk["effective_targets_tcp"]) @ tcp_from_site,
                    np.asarray(chunk["anchor_tcp"]) @ tcp_from_site,
                )
                reference = reference_world_twist(targets, index - chunk["step"], anchor, 0.05)
                independent_reference = generated_twist(targets, index - chunk["step"], anchor, 0.05)
                assert set(reference) == set(independent_reference)
                for key, value in independent_reference.items():
                    if isinstance(value, np.ndarray):
                        np.testing.assert_allclose(reference[key], value, atol=1e-10, rtol=0)
                    else:
                        assert reference[key] == value
                np.testing.assert_allclose(
                    reference["velocity"], original["site_velocity_reference"]["velocity"], atol=1e-9, rtol=0
                )
                desired = np.asarray(original["desired_site"])
                bias = history.current(pose, twist)
                expected_bias = np.zeros(12)
                if independent_last is not None:
                    old_pose, prediction = independent_last
                    expected_bias = np.r_[
                        pose[:3, 3] - old_pose[:3, 3] - prediction[:3],
                        (
                            Rotation.from_matrix(pose[:3, :3] @ old_pose[:3, :3].T)
                            * Rotation.from_rotvec(prediction[3:6]).inv()
                        ).as_rotvec(),
                        0.05 * (twist - prediction[6:]),
                    ]
                np.testing.assert_allclose(bias, expected_bias, atol=1e-9, rtol=0)
                solver_model = FullCoupledPoseModel(mass, jacobian, contract["kp"], contract["kd"])
                solution = serialized(
                    solve_pose(
                        solver_model,
                        pose,
                        desired,
                        twist,
                        reference["twist"],
                        previous,
                        previous_goal,
                        contract,
                        innovation=bias,
                    )
                )
                arguments = (
                    mass,
                    jacobian,
                    contract,
                    pose,
                    desired,
                    twist,
                    reference["twist"],
                    previous,
                    previous_goal,
                    bias,
                )
                certificate = verify_pose_solution(solution, *arguments)
                if index == 0:
                    for path in (
                        ("command",),
                        ("objective",),
                        ("nominal_response",),
                        ("predicted_pose",),
                        ("innovation",),
                        ("mode_candidates", 0, "gradient"),
                    ):
                        broken = copy.deepcopy(solution)
                        obj = broken
                        for key in path[:-1]:
                            obj = obj[key]
                        value = np.asarray(obj[path[-1]], dtype=float).copy()
                        value.flat[0] += 0.02
                        obj[path[-1]] = value.tolist()
                        try:
                            verify_pose_solution(broken, *arguments)
                        except AssertionError:
                            rejected.append("/".join(map(str, path)))
                        else:
                            raise AssertionError(f"Corrupt full-pose solution accepted: {path}")
                command = np.asarray(solution["command"])
                actual_goal, relative = rotation_goal_from_command(command, interface, pose[:3, :3], previous_goal)
                assert (solution["goal_mode"] == "latch") == bool(np.all(interface.decode(command)[3:] == 0))
                continuous_goal = (
                    previous_goal
                    if solution["goal_mode"] == "latch"
                    else Rotation.from_rotvec(interface.decode(command)[3:]).as_matrix() @ pose[:3, :3]
                )
                np.testing.assert_allclose(actual_goal, continuous_goal, atol=1e-6, rtol=0)
                actual_model = build_response(mass, jacobian, contract["kp"], contract["kd"], relative)
                commit = history.commit_executed(pose, twist, actual_model, command[:3], contract)
                independent_actual = independent_response(mass, jacobian, contract["kp"], contract["kd"], relative)
                actual_nominal = (
                    independent_actual["initial_velocity_response"] @ twist
                    + independent_actual["goal_forcing_response"]
                    @ np.r_[interface.decode(command)[:3], independent_actual["rotation_forcing"]]
                )
                np.testing.assert_allclose(commit["nominal_response"], actual_nominal, atol=1e-8, rtol=0)
                independent_last = pose.copy(), actual_nominal.copy()
                next_pose, next_twist, *_ = endpoint(index + 1)
                after = history.observe_completed(next_pose, next_twist)
                difference = np.abs(np.asarray(solution["nominal_response"]) - commit["nominal_response"])
                result = serialized(
                    dict(
                        step=index,
                        solution=solution,
                        reference_twist=reference,
                        previous_goal=previous_goal.copy(),
                        actual_goal=actual_goal,
                        actual_nominal_commit=commit,
                        innovation_before=bias,
                        innovation_after=after,
                        continuous_native_goal_max_difference=float(np.max(np.abs(actual_goal - continuous_goal))),
                        continuous_native_response_block_max=[
                            float(np.max(difference[part]))
                            for part in (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12))
                        ],
                        **certificate,
                    )
                )
                log.write(json.dumps(result, allow_nan=False) + "\n")
                log.flush()
                results.append(result)
                previous, previous_goal, completed = command.copy(), actual_goal.copy(), index + 1
        assert completed == history.completed_periods == 164 and len(rejected) == 6
        e.read(root / "reconstructed_steps.jsonl")
        e.recheck()
        publish(
            root / "independent_verified.json",
            dict(
                complete=True,
                rows=completed,
                independent_full_pose_inverse_and_native_history=True,
                corruptions_rejected=rejected,
                max_independent_stationarity=max(row["max_independent_stationarity"] for row in results),
                maximum_native_goal_difference=max(row["continuous_native_goal_max_difference"] for row in results),
                maximum_native_response_block_difference=np.max(
                    [row["continuous_native_response_block_max"] for row in results], axis=0
                ).tolist(),
                source_sha256=e.hashes,
                seed=42,
                physics_steps=0,
                new_fits=0,
                model_calls=0,
                evidence_level="recorded robot endpoints;reconstructed command/controller goal cache and history;no alternate physical trajectory",
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
