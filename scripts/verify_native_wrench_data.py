"""103 independently verify replay endpoints and actual raw/clipped/ctrl trace."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once

from s2a.owned_receipt_contract import normal_receipts
from s2a.xml_site_kinematics import XMLSiteKinematics

ROOT = Path("outputs/native_wrench_data_20261025")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", type=int, choices=range(30), required=True)
    args = p.parse_args()
    e = Evidence()
    e.read(__file__)
    folder = ROOT / f"group{args.group:02d}"
    receipts = ROOT / "orchestration" / f"group{args.group:02d}"
    normal_receipts(e, receipts / "launch.json", receipts / "exit.json", receipts / "release.json", format="durable")
    c = e.json(folder / "collection_result.json")
    assert c["complete"] and c["group"] == args.group and c["inner_records"] == 25
    for path, signature in c["source_sha256"].items():
        e.check(path, signature)
    for key in ("arrays_ref", "trace_ref", "baseline_ref", "wrench_ref", "native_cache_ref"):
        ref = c[key]
        e.check(ref["path"], ref["sha256"])
    with (
        np.load(c["arrays_ref"]["path"], allow_pickle=False) as actual,
        np.load(c["baseline_ref"]["path"], allow_pickle=False) as old,
    ):
        for key in actual.files:
            if key == "source_twist":
                with np.load(c["native_cache_ref"]["path"], allow_pickle=False) as cache:
                    expected_twist = np.vstack([cache["x_site_twist"], cache["y_twist"][-1]])
                np.testing.assert_allclose(actual[key], expected_twist, rtol=0, atol=1e-9)
            elif key == "sent_commands":
                np.testing.assert_array_equal(actual[key], old[key])
            else:
                np.testing.assert_allclose(actual[key], old[key], rtol=0, atol=1e-9)
        assert len(actual["sent_commands"]) == c["physics_steps"]
        nqnv = actual["sim_states"].copy()
    registration = e.json("configs/native_wrench_data_20261025.json")
    profile = registration["profiles"][args.group]
    assert c["native_index"] == profile["index"] == args.group and c["split"] == profile["split"]
    assert c["baseline_ref"] == profile["trajectory_ref"] and c["native_cache_ref"] == profile["cache_ref"]
    rows = e.json(c["trace_ref"]["path"])["records"]
    assert len(rows) == 25
    np.testing.assert_allclose(np.diff([r["time"] for r in rows]), 0.002, rtol=0, atol=1e-10)
    np.testing.assert_allclose(rows[0]["time"], nqnv[-2, 0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(rows[-1]["time"] + 0.002, nqnv[-1, 0], rtol=0, atol=1e-10)
    saturated = 0
    for row in rows:
        raw, clipped, ctrl, lo, hi = (
            np.asarray(row[k]) for k in ("raw_torque", "clipped_torque", "ctrl", "low", "high")
        )
        assert raw.shape == clipped.shape == ctrl.shape == lo.shape == hi.shape == (7,)
        assert all(np.isfinite(v).all() for v in (raw, clipped, ctrl, lo, hi))
        np.testing.assert_array_equal(clipped, np.minimum(hi, np.maximum(lo, raw)))
        np.testing.assert_array_equal(ctrl, clipped)
        saturated += int(np.any((raw < lo) | (raw > hi)))
    samples = e.json(c["wrench_ref"]["path"])["samples"]
    assert len(samples) == c["wrench_endpoints"] == c["physics_steps"] + 1
    xml_bytes = e.read(folder / "model.xml")
    xml = ET.fromstring(xml_bytes)
    sensor_kinematics = XMLSiteKinematics(xml_bytes, samples[0]["sensor_site"])
    sensor_nodes = {node.get("name"): node for node in xml.findall("sensor/*")}
    for index, sample in enumerate(samples):
        assert sample["readonly_verified"]
        assert sample["sensor_names"] == ["gripper0_force_ee", "gripper0_torque_ee"]
        for name, tag in zip(sample["sensor_names"], ("force", "torque"), strict=True):
            assert sensor_nodes[name].tag == tag and sensor_nodes[name].get("site") == sample["sensor_site"]
        local = np.array(sample["local"])
        pose = np.array(sample["sensor_pose"])
        point = np.array(sample["control_site_position"])
        actual_wrench = np.array(sample["world_at_control_site"])
        independent_pose, _ = sensor_kinematics.endpoint(nqnv[index])
        np.testing.assert_allclose(pose, independent_pose, rtol=0, atol=1e-9)
        assert sample["sensor_types"] == [4, 5] and sample["sensor_addresses"] == [0, 3]
        assert (
            local.shape == actual_wrench.shape == (6,) and np.isfinite(local).all() and np.isfinite(actual_wrench).all()
        )
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), rtol=0, atol=1e-10)
        assert abs(np.linalg.det(pose[:3, :3]) - 1) < 1e-10
        f = pose[:3, :3] @ local[:3]
        t = pose[:3, :3] @ local[3:] + np.cross(pose[:3, 3] - point, f)
        np.testing.assert_allclose(actual_wrench, np.r_[f, t], rtol=0, atol=1e-10)
        np.testing.assert_allclose(sample["time"], nqnv[index, 0], rtol=0, atol=1e-10)
    with np.load(c["arrays_ref"]["path"], allow_pickle=False) as actual:
        np.testing.assert_allclose(
            [s["control_site_position"] for s in samples], actual["controller_site_pose"][:, :3, 3], rtol=0, atol=1e-10
        )
    flags = sorted(set(r["uncoupling"] for r in rows))
    assert len(flags) == 1
    e.recheck()
    write_once(
        folder / "independent_verified.json",
        dict(
            complete=True,
            group=args.group,
            prefix_matches=True,
            native_index=c["native_index"],
            split=c["split"],
            current_wrench_contract_verified=True,
            sensor_frame_independent_xml_verified=True,
            wrench_endpoints=len(samples),
            actual_uncoupling=flags[0],
            saturated_inner_calls=saturated,
            inner_calls=25,
            physics_steps=c["physics_steps"],
            source_sha256=e.hashes,
            model_calls=0,
            new_fits=0,
            fresh_cases=0,
            control_qualification=False,
        ),
    )
    print(dict(group=args.group, uncoupling=flags[0], saturated=saturated, inner_calls=25), flush=True)


if __name__ == "__main__":
    main()
