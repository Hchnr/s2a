"""227 qualify completed instance records after independent IDs and M/J checks."""

import hashlib

import mujoco
import numpy as np
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    root = REPO / "outputs/actual_controller_coupling_probe_verified_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (__file__, "plans/v0.0.1/227_controller_probe_identity_verification_20261026.md"):
        e.read(path)
    original = e.json("outputs/actual_controller_coupling_probe_checked_20261026/failure.json")
    assert original["complete"] is False and original["error_type"] == "AssertionError" and original["error"] == ""
    for path, signature in original["source_sha256"].items():
        e.check(path, signature)
    source = e.read("scripts/probe_actual_controller_coupling_checked.py")
    assert b'assert set(row["task_id"] for row in results) == {0, 1, 2}' in source
    assert [(r["case"], r["step"]) for r in original["instances"]] == [(15, 550), (6, 220), (14, 60)]
    calls = original["simulation_entry_calls"]
    assert calls == dict(mj_step=0, mj_step1=6, mj_step2=0)
    metadata = e.json("outputs/controller_probe_step_semantics_20261026/source.json")
    content = e.read("outputs/controller_probe_step_semantics_20261026/engine_forward_2.3.7_exact.c")
    blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    assert blob == metadata["blob_sha"] == "61906d629a8d3e071a922977a4afc86bd0126a3d"
    e.read("outputs/controller_probe_step_semantics_20261026/snapshot_record.json")
    registration = e.json("configs/preview_control_registered_20261026.json")
    results = []
    for saved in original["instances"]:
        index, step = saved["case"], saved["step"]
        directory = REPO / f"outputs/preview_control_20261026/case{index:02d}"
        case = next((directory / "simulation").glob("*/*"))
        manifest = e.json(case / "manifest.json")
        record = registration["profiles"][index]["record"]
        assert manifest["record"] == record
        assert saved["task_id"] == manifest["record"]["task_id"] == manifest["runtime_config"]["task_id"]
        assert (
            saved["actual_uncoupling"] is True
            and saved["actual_use_delta"] is True
            and saved["actual_impedance_mode"] == "fixed"
        )
        assert saved["state_unchanged"] and saved["max_mass_difference"] == saved["max_jacobian_difference"] == 0
        xml = case / "model.xml"
        e.check(xml, manifest["initial_model_xml_sha256"])
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        ref = e.json(directory / "collection_result.json")["arrays_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as archive:
            state = archive["sim_states"][step].copy()
            pose = archive["controller_site_pose"][step].copy()
        assert len(state) == 1 + model.nq + model.nv
        data.time = state[0]
        data.qpos[:] = state[1 : 1 + model.nq]
        data.qvel[:] = state[1 + model.nq :]
        mujoco.mj_forward(model, data)
        contract = e.json(case / "controller.json")["contract"]
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, contract["controller_site"])
        assert sid >= 0
        np.testing.assert_allclose(data.site_xpos[sid], pose[:3, 3], atol=1e-9, rtol=0)
        np.testing.assert_allclose(data.site_xmat[sid].reshape(3, 3), pose[:3, :3], atol=1e-9, rtol=0)
        dofs = []
        for j in range(1, 8):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{j}")
            assert jid >= 0
            dofs.append(model.jnt_dofadr[jid])
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, sid)
        mass_difference = float(np.max(np.abs(mass[np.ix_(dofs, dofs)] - saved["mass"])))
        jacobian_difference = float(np.max(np.abs(np.vstack([jp, jr])[:, dofs] - saved["jacobian"])))
        assert mass_difference < 1e-9 and jacobian_difference < 1e-9
        np.testing.assert_array_equal(np.r_[data.time, data.qpos, data.qvel], state)
        results.append(
            dict(
                case=index,
                step=step,
                record=record["name"],
                task_id=saved["task_id"],
                actual_uncoupling=saved["actual_uncoupling"],
                independent_mass_difference=mass_difference,
                independent_jacobian_difference=jacobian_difference,
            )
        )
    assert len({r["task_id"] for r in results}) == 3 and {r["task_id"] for r in results} == {8, 9, 6}
    e.recheck()
    root.mkdir()
    publish(
        root / "verified.json",
        dict(
            complete=True,
            instances=results,
            actual_reconstructed_instance_fields_verified=True,
            original_producer_status="postvalidation task_id assertion failed; preserved",
            original_instance_collection_completed=True,
            original_simulation_entry_calls=calls,
            physics_steps=0,
            new_instances_created=0,
            new_fits=0,
            model_calls=0,
            upstream_blob_sha=blob,
            source_sha256=e.hashes,
            seed=42,
            historical_process_fields_directly_recorded=False,
            control_qualified=False,
        ),
    )
    print(
        "227 IDs8/9/6 and saved actual uncoupling/M/J independently verified;original failure preserved;no integration"
    )


if __name__ == "__main__":
    main()
