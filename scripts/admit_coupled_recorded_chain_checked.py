"""230 full current-matrix inverse/history admission on recorded states only."""

import copy
import ctypes
import json
import os

import numpy as np
from audit_wam_position import Evidence
from current_coupled_independent import response, verify_inverse
from run_matched_oracle_case import CONTROL, CONTROL_SHA
from scipy.spatial.transform import Rotation

from s2a.archived_forward_state import verify_forward_state
from s2a.artifacts import REPO, sha256
from s2a.coupled_response_history import CoupledResponseHistory
from s2a.current_coupled_response import build_response, solve_translation
from s2a.current_coupled_state import rotation_goal_from_command
from s2a.durable_stage_supervisor import publish
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface


def main():
    strict_native_environment()
    assert sha256(CONTROL) == CONTROL_SHA
    guard = json.loads(CONTROL.read_text())["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    root = REPO / "outputs/current_coupled_recorded_chain_checked_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "src/s2a/current_coupled_response.py",
        "src/s2a/archived_forward_state.py",
        "tests/test_archived_forward_state.py",
        "plans/v0.0.1/233_archival_quaternion_normalization_20261026.md",
        "outputs/current_coupled_recorded_chain_20261026/failure.json",
        "src/s2a/coupled_response_history.py",
        "src/s2a/current_coupled_state.py",
        "scripts/current_coupled_independent.py",
        "plans/v0.0.1/230_coupled_recorded_chain_20261026.md",
    ):
        e.read(path)
    proof = e.json("outputs/current_coupled_reader_admission_20261026/verified.json")
    assert proof["complete"] and proof["current_dynamics_reader_verified"]
    for path, signature in proof["source_sha256"].items():
        e.check(path, signature)
    closure = e.json("outputs/preview_control_serialized_20261026/closure_verified.json")
    assert closure["complete"] and closure["owner_normal_wait_verified"]
    for path, signature in closure["source_sha256"].items():
        e.check(path, signature)
    directory = REPO / "outputs/preview_control_20261026/case00"
    case = next((directory / "simulation").glob("*/*"))
    rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
    assert len(rows) == 164
    contract = e.json(case / "controller.json")["contract"]
    interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    ref = e.json(directory / "collection_result.json")["arrays_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as archive:
        states = archive["sim_states"].copy()
        poses = archive["controller_site_pose"].copy()
        goals = archive["controller_goal_ori"].copy()
    manifest = e.json(case / "manifest.json")
    ctypes.CDLL(manifest["runtime_config"]["osmesa_library"], mode=ctypes.RTLD_GLOBAL)
    import mujoco

    xml = case / "model.xml"
    e.check(xml, manifest["initial_model_xml_sha256"])
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
    assert sid >= 0
    qids, dofs = [], []
    for joint in range(1, 8):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{joint}")
        assert jid >= 0
        qids.append(model.jnt_qposadr[jid])
        dofs.append(model.jnt_dofadr[jid])

    quaternion_segments = []
    for joint in range(model.njnt):
        kind = int(model.jnt_type[joint])
        if kind in (0, 1):
            quaternion_segments.append(
                dict(
                    joint=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
                    qpos_start=int(model.jnt_qposadr[joint]) + (3 if kind == 0 else 0),
                )
            )
    normalization_checks = []

    def endpoint(index):
        state = states[index]
        data.time = state[0]
        data.qpos[:] = state[1 : 1 + model.nq]
        data.qvel[:] = state[1 + model.nq :]
        mujoco.mj_forward(model, data)
        pose = np.eye(4)
        pose[:3, 3] = data.site_xpos[sid]
        pose[:3, :3] = data.site_xmat[sid].reshape(3, 3)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, sid)
        jacobian = np.vstack([jp, jr])
        np.testing.assert_allclose(pose, poses[index], atol=1e-9, rtol=0)
        changes = verify_forward_state(
            state, np.r_[data.time, data.qpos, data.qvel], nq=model.nq, quaternion_segments=quaternion_segments
        )
        if changes:
            normalization_checks.append(dict(endpoint_index=index, changes=changes))
        return pose, jacobian @ data.qvel, jacobian[:, dofs].copy()

    root.mkdir()
    os.environ.update(NUMBA_CACHE_DIR=str(root / "numba_cache"), MPLCONFIGDIR=str(root / "matplotlib"))
    history = CoupledResponseHistory()
    previous = np.zeros(3)
    independent_last = None
    results, rejected = [], []
    try:
        for index, row in enumerate(rows):
            pose, twist, jacobian = endpoint(index)
            packet = {key: np.asarray(value) for key, value in row["qualified_forward_input"].items()}
            np.testing.assert_allclose(data.qpos[qids], packet["x_joint_position"][0], atol=1e-12, rtol=0)
            np.testing.assert_allclose(data.qvel[dofs], packet["x_joint_velocity"][0], atol=1e-12, rtol=0)
            np.testing.assert_allclose(twist, packet["x_site_twist"][0], atol=1e-8, rtol=0)
            mass = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, mass, data.qM)
            arm_mass = mass[np.ix_(dofs, dofs)]
            goal, relative = rotation_goal_from_command(
                np.asarray(row["command"])[:6], interface, pose[:3, :3], packet["x_previous_goal_ori"][0]
            )
            np.testing.assert_allclose(goal, goals[index + 1], atol=1e-8, rtol=0)
            current = build_response(arm_mass, jacobian, contract["kp"], contract["kd"], relative)
            independent = response(arm_mass, jacobian, contract["kp"], contract["kd"], relative)
            for key in independent:
                np.testing.assert_allclose(current[key], independent[key], atol=1e-8, rtol=0)
            observed = history.current(pose, twist)
            expected = np.zeros(12)
            if independent_last is not None:
                old_pose, prediction = independent_last
                rotation = (
                    Rotation.from_matrix(pose[:3, :3] @ old_pose[:3, :3].T)
                    * Rotation.from_rotvec(prediction[3:6]).inv()
                ).as_rotvec()
                expected = np.r_[
                    pose[:3, 3] - old_pose[:3, 3] - prediction[:3], rotation, 0.05 * (twist - prediction[6:])
                ]
            np.testing.assert_allclose(observed, expected, atol=1e-8, rtol=0)
            innovation = np.vstack([observed[:3], observed[6:9] / 0.05])
            target = np.asarray(row["desired_site"])[:3, 3]
            velocity = np.asarray(row["site_velocity_reference"]["velocity"])
            solution = solve_translation(
                pose[:3, 3], twist, target, velocity, previous, contract, current, innovation=innovation
            )
            saved = json.loads(json.dumps(solution, default=lambda value: value.tolist(), allow_nan=False))
            arguments = (pose[:3, 3], twist, target, velocity, previous, contract, independent, innovation)
            certificate = verify_inverse(saved, *arguments)
            if index == 0:
                for key in (
                    "command",
                    "physical_delta",
                    "nominal_response",
                    "predicted_response",
                    "objective",
                    "innovation",
                ):
                    broken = copy.deepcopy(saved)
                    if key == "objective":
                        broken[key] += 0.02
                    else:
                        changed = np.asarray(broken[key], dtype=float).copy()
                        changed.flat[0] += 0.02
                        broken[key] = changed.tolist()
                    try:
                        verify_inverse(broken, *arguments)
                    except AssertionError:
                        rejected.append(key)
                    else:
                        raise AssertionError(f"Corrupt coupled solution accepted: {key}")
            commit = history.commit_executed(pose, twist, current, np.asarray(saved["command"]), contract)
            nominal = (
                independent["initial_velocity_response"] @ twist
                + independent["goal_forcing_response"] @ np.r_[saved["physical_delta"], independent["rotation_forcing"]]
            )
            np.testing.assert_allclose(commit["nominal_response"], nominal, atol=1e-8, rtol=0)
            independent_last = pose.copy(), nominal.copy()
            next_pose, next_twist, _ = endpoint(index + 1)
            history.observe_completed(next_pose, next_twist)
            previous = np.asarray(saved["command"])
            results.append(dict(step=index, completed_periods=history.completed_periods, solution=saved, **certificate))
        assert history.completed_periods == 164 and len(rejected) == 6
        e.recheck()
        publish(
            root / "independent_verified.json",
            dict(
                complete=True,
                rows=164,
                independent_coupled_inverse_and_history=True,
                json_roundtrip_rows=164,
                archival_quaternion_normalization_checks=normalization_checks,
                normalized_unique_endpoints=sorted({check["endpoint_index"] for check in normalization_checks}),
                corruptions_rejected=rejected,
                max_stationarity_residual=max(row["max_stationarity_residual"] for row in results),
                results=results,
                source_sha256=e.hashes,
                physics_steps=0,
                new_fits=0,
                model_calls=0,
                seed=42,
                evidence_level="recorded robot states with reconstructed commands/history; no alternate physical trajectory",
                control_qualified=False,
            ),
        )
    except Exception as error:
        publish(
            root / "failure.json",
            dict(
                complete=False,
                checked_rows=len(results),
                error_type=type(error).__name__,
                error=str(error),
                source_sha256=e.hashes,
                physics_steps=0,
                automatic_retry=False,
            ),
        )
        raise
    print("230164states coupled response/inverse/history and JSON KKT verified;6 corruptions rejected;no physics")


if __name__ == "__main__":
    main()
