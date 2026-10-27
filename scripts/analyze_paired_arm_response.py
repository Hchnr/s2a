"""086 independently bind complete anchor groups and summarize measured local gains."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from scipy.linalg import expm

from s2a.paired_response_analysis import axis_response

ROOT = Path("outputs/paired_arm_response_20261025")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verified-prefix",
        action="store_true",
        help="Only complete verified anchor groups; not full data qualification",
    )
    args = parser.parse_args()
    e = Evidence()
    e.read(__file__)
    e.read("src/s2a/paired_response_analysis.py")
    config = e.json("configs/paired_arm_response_20261025.json")
    if not args.verified_prefix:
        closure = e.json(ROOT / "closure_verified.json")
        assert (
            closure["complete"]
            and closure["data_qualified"]
            and closure["verified_cases"] == 42
            and closure["owner_normal_wait_verified"]
        )
        ref = closure["summary_ref"]
        e.check(ref["path"], ref["sha256"])
        e.json(ref["path"])
    for path, signature in e.json(ROOT / "source_sha256.json").items():
        e.check(path, signature)
    groups = []
    for group in range(6):
        indices = list(range(group * 7, group * 7 + 7))
        if not all((ROOT / "verification" / f"case{i:02d}" / "exit.json").exists() for i in indices):
            assert args.verified_prefix
            continue
        packets = []
        contract = None
        for i in indices:
            folder = ROOT / f"case{i:02d}"
            trial = config["trials"][i]
            assert trial["anchor_group"] == group and trial["anchor_cycle"] + 1 == trial["physics_steps"]
            exit_record = e.json(ROOT / "verification" / f"case{i:02d}" / "exit.json")
            assert exit_record["returncode"] == 0
            verified = e.json(folder / "independent_verified.json")
            assert verified["complete"] and verified["mechanical_verified"] and verified["prefix_native_match"]
            assert verified["source_twist_independent_xml_verified"] and verified["current_images_verified"]
            result = e.json(folder / "collection_result.json")
            ref = result["arrays_ref"]
            e.check(ref["path"], ref["sha256"])
            if contract is None:
                contract = result["controller_contract"]
            else:
                assert contract == result["controller_contract"]
            with np.load(ref["path"], allow_pickle=False) as data:
                k = trial["anchor_cycle"]
                packet = dict(
                    pose=data["controller_site_pose"][-1].copy(),
                    twist=data["source_twist"][-1].copy(),
                    command=data["sent_commands"][-1].copy(),
                    before=data["controller_site_pose"][k].copy(),
                    before_twist=data["source_twist"][k].copy(),
                    state_before=data["sim_states"][k].copy(),
                    goal=data["controller_goal_pos"][-1].copy(),
                    goal_rotation=data["controller_goal_ori"][-1].copy(),
                )
            if packets:
                for key, tol in [
                    ("before", 1e-10),
                    ("before_twist", 1e-9),
                    ("state_before", 1e-10),
                    ("goal_rotation", 1e-9),
                ]:
                    np.testing.assert_allclose(packet[key], packets[0][key], rtol=0, atol=tol)
            lo, hi = np.asarray(contract["input_min"]), np.asarray(contract["input_max"])
            outlo, outhi = np.asarray(contract["output_min"]), np.asarray(contract["output_max"])
            delta = outlo + (packet["command"][:6] - lo) * (outhi - outlo) / (hi - lo)
            np.testing.assert_allclose(packet["goal"], packet["before"][:3, 3] + delta[:3], rtol=0, atol=1e-8)
            packets.append(packet)
        assert e.json(ROOT / f"case{indices[0]:02d}" / "independent_verified.json")["baseline_native_match"]
        scale = (outhi[:3] - outlo[:3]) / (hi[:3] - lo[:3])
        axes = [axis_response(packets[0], packets[1 + 2 * a], packets[2 + 2 * a], a, scale[a]) for a in range(3)]
        matrices = {
            key: np.column_stack([a["central"]["per_normalized_input"][key] for a in axes])
            for key in ("position", "linear_velocity", "so3_increment", "angular_velocity")
        }
        dt = 1 / contract["control_freq"]
        assert dt == 0.05
        kp, kd = np.asarray(contract["kp"])[:3], np.asarray(contract["kd"])[:3]
        transitions = np.array([expm(np.array([[0.0, 1.0], [-p, -d]]) * dt) for p, d in zip(kp, kd, strict=True)])
        nominal_position = np.diag((1 - transitions[:, 0, 0]) * scale)
        nominal_velocity = np.diag(-transitions[:, 1, 0] * scale)
        pos = matrices["position"]
        off = pos - np.diag(np.diag(pos))
        singular = np.linalg.svd(pos, compute_uv=False)
        groups.append(
            dict(
                anchor_group=group,
                profile_index=config["trials"][indices[0]]["profile_index"],
                phase=config["trials"][indices[0]]["phase"],
                anchor_cycle=config["trials"][indices[0]]["anchor_cycle"],
                axes=axes,
                central_per_normalized_input=matrices,
                nominal_position_per_normalized_input=nominal_position,
                nominal_velocity_per_normalized_input=nominal_velocity,
                position_same_axis_gain_ratio=np.diag(pos) / np.diag(nominal_position),
                position_cross_axis_frobenius_m_per_input=float(np.linalg.norm(off)),
                position_singular_values_m_per_input=singular,
                rotation_coupling_frobenius_rad_per_input=float(np.linalg.norm(matrices["so3_increment"])),
            )
        )
    assert groups
    e.recheck()
    result = dict(
        complete=not args.verified_prefix,
        verified_anchor_groups=len(groups),
        groups=groups,
        full_data_qualification=not args.verified_prefix,
        task_success_qualification=False,
        source_sha256=e.hashes,
        physics_steps=0,
        model_calls=0,
        new_fits=0,
        fresh_cases=0,
        evidence="one-period finite command differences at six registered fit anchors; not global dynamics or closed-loop stability",
        pose_frame="actual control site; world SO3 increments; fixed rigid TCP rotation cancels in the relative world rotation",
    )
    out = Path("outputs/paired_arm_response_analysis_20261025")
    out.mkdir(exist_ok=True)
    name = ("verified_groups_" if args.verified_prefix else "complete_") + f"{len(groups):02d}.json"
    with (out / name).open("x") as stream:
        json.dump(result, stream, indent=2, default=lambda value: value.tolist(), allow_nan=False)
        stream.write("\n")
    for group in groups:
        print(
            "group",
            group["anchor_group"],
            "same-axis ratios",
            group["position_same_axis_gain_ratio"].tolist(),
            "rotation coupling",
            group["rotation_coupling_frobenius_rad_per_input"],
        )


if __name__ == "__main__":
    main()
