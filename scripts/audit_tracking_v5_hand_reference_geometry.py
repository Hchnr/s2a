"""Measure frozen joint prediction errors against independently replayed inner states."""

import json
import os
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.convex_mesh_box_diagnostic import axis_clearance
from s2a.geometry import inverse
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_hand_reference_geometry_20261028"
    require(not root.exists(), "No overwrite/retry")
    owner_ref = reference(REPO / "outputs/tracking_v5_causal_inner_continuation_check_execution_20261028/result.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Closed independent full-inner replay")
    check_ref = owner["result_ref"]
    require(read(check_ref)["complete"] is read(check_ref)["passed"] is True, "Verified all-inner states")
    producer_ref = reference(REPO / "outputs/tracking_v5_causal_inner_continuation_20261028/result.json")
    producer = read(producer_ref)
    reg = read(producer["registration_ref"])
    worker = read(reg["worker_ref"])
    folder = Path(worker["physical_case_dir"])
    control_ref = reference(folder / "controller.json")
    control = read(control_ref)
    contract = control["contract"]
    manifest = read(reg["manifest_ref"])
    tool_ref = reference(Path(manifest["runtime_config"]["tool_contract"]))
    site_from_tcp = inverse(np.asarray(control["source_from_site"])) @ np.asarray(read(tool_ref)["source_from_tcp"])
    numeric_ref = reference(REPO / "outputs/tracking_v5_velocity_braking_numeric_20261028/result.json")
    sources = {
        ref["path"]: ref["sha256"]
        for ref in [
            reference(Path(__file__)),
            *[reference(REPO / "src/s2a" / name) for name in ("convex_mesh_box_diagnostic.py", "geometry.py")],
        ]
    }
    registration = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            independent_owner_ref=owner_ref,
            independent_ref=check_ref,
            producer_ref=producer_ref,
            numeric_ref=numeric_ref,
            control_ref=control_ref,
            xml_ref=reference(folder / "model.xml"),
            seed=42,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            rendering_policy="XML-only CPU mj_forward; no renderer/context; OSMesa forcing unset before MuJoCo import",
            envelope="Per-joint running maximum of earlier completed-period absolute free-model residual; first period has no bound",
            scope="All actually executed periods; independent native-state causal-history/plan audit; mj_forward only, no task qualification",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    # XML-only CPU dynamics: no OpenGL context or renderer is constructed.
    os.environ.pop("MUJOCO_GL", None)
    os.environ.pop("PYOPENGL_PLATFORM", None)
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(folder / "model.xml"))
    data = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1, 8)]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
    require(sid >= 0 and all(i >= 0 for i in joints), "Actual robot joints/SITE")
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        branches=[],
    )
    try:
        branch = next(v for v in producer["branches"] if v["branch"] == "causal_error")
        with np.load(checked(branch["states_ref"]), allow_pickle=False) as archive:
            states = archive["sim_states"].copy()
        chunks = read(reg["chunks_ref"])
        targets = np.asarray(chunks[46]["effective_targets_tcp"])
        hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "gripper0_hand_collision")
        stove = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "flat_stove_1_collision_burner")
        require(hand >= 0 and stove >= 0, "Logged collision pair exists")
        require(
            model.geom_type[hand] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_type[stove] == mujoco.mjtGeom.mjGEOM_BOX,
            "Actual mesh and box",
        )
        mesh = model.geom_dataid[hand]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start : start + count].copy()

        def geom_pose(gid):
            value = np.eye(4)
            value[:3, :3] = data.geom_xmat[gid].reshape(3, 3)
            value[:3, 3] = data.geom_xpos[gid]
            return value

        def evaluate(hand_pose, stove_pose):
            points = vertices @ hand_pose[:3, :3].T + hand_pose[:3, 3]
            return axis_clearance(points, stove_pose, model.geom_size[stove])

        actual_records = []
        tcp_from_hand, stove_anchor = None, None
        for step in (368, 388, 389, 390):
            state = states[step]
            data.time = state[0]
            data.qpos[:] = state[1 : 1 + model.nq]
            data.qvel[:] = state[1 + model.nq :]
            mujoco.mj_forward(model, data)
            site_pose = np.eye(4)
            site_pose[:3, :3] = data.site_xmat[sid].reshape(3, 3)
            site_pose[:3, 3] = data.site_xpos[sid]
            tcp = site_pose @ site_from_tcp
            hand_pose, stove_pose = geom_pose(hand), geom_pose(stove)
            relative = inverse(tcp) @ hand_pose
            if tcp_from_hand is None:
                tcp_from_hand, stove_anchor = relative.copy(), stove_pose.copy()
            np.testing.assert_allclose(relative, tcp_from_hand, atol=1e-9, rtol=0)
            np.testing.assert_allclose(stove_pose, stove_anchor, atol=1e-12, rtol=0)
            native_pair = [
                float(c.dist) for c in data.contact[: data.ncon] if {int(c.geom1), int(c.geom2)} == {hand, stove}
            ]
            actual_records.append(
                dict(
                    endpoint=step,
                    actual_tcp=tcp,
                    hand_pose=hand_pose,
                    axis_test=evaluate(hand_pose, stove_pose),
                    native_pair_distances=native_pair,
                )
            )
        reference_records = [
            dict(
                cursor=i,
                desired_endpoint=369 + i,
                desired_tcp=target,
                axis_test=evaluate(target @ tcp_from_hand, stove_anchor),
            )
            for i, target in enumerate(targets)
        ]
        result.update(
            complete=True,
            passed=True,
            actual_states_ref=branch["states_ref"],
            chunks_ref=reg["chunks_ref"],
            tool_ref=tool_ref,
            selected_pair=dict(hand=hand, stove=stove, mesh_vertices=int(count)),
            tcp_from_hand=tcp_from_hand,
            actual_records=actual_records,
            reference_records=reference_records,
            scope="Offline rigid hand convex hull versus fixed burner box at saved and reference endpoints only; no IK, continuous sweep, other collision, online oracle, physical rollout or task qualification",
        )
        for path, digest in sources.items():
            checked(dict(path=path, sha256=digest))
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
