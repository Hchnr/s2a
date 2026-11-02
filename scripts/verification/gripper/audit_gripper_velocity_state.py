"""Read-only audit of instantaneous gripper velocity saved in native state vectors."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PREPARED = Path(
    "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/preparation/prepared.json"
)
ROOT = Path("outputs/gripper_velocity_state_audit_20261025")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    binding = Path("outputs/libero_env311/lib/python3.11/site-packages/robosuite/utils/binding_utils.py")
    sensor = Path("outputs/libero_env311/lib/python3.11/site-packages/robosuite/robots/single_arm.py")
    paths = [PREPARED, Path(__file__), binding, sensor]
    roster = {}
    for anchor in json.loads(PREPARED.read_text())["training_anchors"]:
        roster.setdefault(anchor["trajectory_ref"]["path"], (anchor["trajectory_ref"], anchor["case_ref"]))
    assert len(roster) == 30
    for trajectory, case in roster.values():
        paths.extend([Path(trajectory["path"]), Path(case["path"])])
    hashes = {str(p): digest(p) for p in paths}
    cases = []
    traces = {}
    for index, (trajectory, case) in enumerate(roster.values()):
        assert digest(trajectory["path"]) == trajectory["sha256"] and digest(case["path"]) == case["sha256"]
        record = json.loads(Path(case["path"]).read_text())["record"]
        xml = ET.fromstring(record["model_xml"])
        nq = nv = 0
        indices = {}
        for node in xml.find("worldbody").iter():
            if node.tag not in ("joint", "freejoint"):
                continue
            kind = "free" if node.tag == "freejoint" else node.get("type", "hinge")
            if node.get("name") in ("gripper0_finger_joint1", "gripper0_finger_joint2"):
                assert kind == "slide"
                indices[node.get("name")] = (nq, nv)
            width, velocity_width = {"free": (7, 6), "ball": (4, 3), "hinge": (1, 1), "slide": (1, 1)}[kind]
            nq += width
            nv += velocity_width
        assert len(indices) == 2
        qids, vids = zip(*(indices[f"gripper0_finger_joint{i}"] for i in (1, 2)), strict=True)
        with np.load(trajectory["path"], allow_pickle=False) as data:
            state = data["sim_states"]
            q = data["opening"].copy()
            recorded_qpos = data["gripper_qpos"]
            times = data["sim_time"]
            assert state.shape == (len(q), 1 + nq + nv)
            np.testing.assert_array_equal(state[:, 0], times)
            qpos = state[:, [1 + i for i in qids]]
            np.testing.assert_array_equal(qpos, recorded_qpos)
            velocity = state[:, [1 + nq + i for i in vids]]
        np.testing.assert_allclose(np.clip((qpos[:, 0] - qpos[:, 1] - 0.001) / 0.079, 0, 1), q, rtol=0, atol=1e-12)
        instantaneous = (velocity[:, 0] - velocity[:, 1]) / 0.079
        previous = np.r_[0.0, np.diff(q)] / 0.05
        interior = (q > 0.05) & (q < 0.95)
        metrics = dict(
            ref=trajectory,
            case_ref=case,
            nq=nq,
            nv=nv,
            joint_indices=indices,
            endpoints=len(q),
            interior_endpoints=int(interior.sum()),
            interior_velocity_difference_rms_per_s=float(
                np.sqrt(np.mean((instantaneous[interior] - previous[interior]) ** 2))
            ),
        )
        if record["name"] == "task1_native_demo_39":
            metrics["opening_failure_window"] = [
                dict(
                    endpoint=k,
                    opening=float(q[k]),
                    instantaneous_delta_equivalent=float(0.05 * instantaneous[k]),
                    previous_opening_delta=float(0.05 * previous[k]),
                    actual_next_opening_delta=float(q[k + 1] - q[k]),
                )
                for k in range(155, 163)
            ]
            print(json.dumps(metrics, indent=2))
        cases.append(metrics)
        traces[f"case{index:02d}_velocity"] = instantaneous
        traces[f"case{index:02d}_previous_difference_velocity"] = previous
    assert all(digest(p) == s for p, s in hashes.items())
    ROOT.mkdir(exist_ok=False)
    np.savez(ROOT / "velocity_traces.npz", **traces)
    output = dict(
        complete=True,
        evidence="saved current qvel parsed using verified state layout; not a model fit or physical rollout",
        source_sha256=hashes,
        cases=cases,
        trace_sha256=digest(ROOT / "velocity_traces.npz"),
        new_fits=0,
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    (ROOT / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print("audited30 trajectories; all saved gripper qpos and opening indices verified")


if __name__ == "__main__":
    main()
