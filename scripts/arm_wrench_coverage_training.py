"""107 current-wrench producer-side additional63 equations, gated on full106 data closure."""

from pathlib import Path

import numpy as np
from arm_wrench_data import attach_current_wrench
from arm_wrench_independent import INPUTS
from scipy.spatial.transform import Rotation

from s2a.wrench_conditioned_arm_forward import design, nominal, with_history


def contrast_rows(config, e, parameters):
    ref = config["coverage_closure_ref"]
    e.check(ref["path"], ref["sha256"])
    closure = e.json(config["coverage_closure_ref"]["path"])
    assert closure["complete"] and closure["data_qualified"]
    assert closure["verified_cases"] == 147 and closure["owner_normal_wait_verified"]
    ref = closure["summary_ref"]
    e.check(ref["path"], ref["sha256"])
    summary = e.json(ref["path"])
    registration_path = Path("configs/wrench_coverage_interventions_20261025.json")
    e.check(registration_path, closure["source_sha256"][str(registration_path.resolve())])
    paired_config = e.json("configs/wrench_coverage_interventions_20261025.json")
    source = e.json(config["data_result_ref"]["path"])
    assert len(paired_config["profiles"]) == 21 and paired_config["all_new_data_split"] == "fit"
    x = []
    y = []
    base = []
    records = []
    for group in range(21):
        native_profile = paired_config["profiles"][group]
        entry = next(
            p for p in source["trajectories"] if p["source"]["sha256"] == native_profile["trajectory_ref"]["sha256"]
        )
        assert entry["split"] == "fit"
        anchor = paired_config["trials"][group * 7]["anchor_cycle"]
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as cache:
            full = with_history({k: cache[k].copy() for k in cache.files}, parameters)
            full = attach_current_wrench(full, entry, config, e)
            condition = {k: full[k][anchor : anchor + 1].copy() for k in INPUTS}
        current = condition["x_site_pose"][0]
        packets = []
        targets = []
        refs = []
        contract = None
        for branch in range(7):
            index = group * 7 + branch
            ref = summary["cases"][index]["arrays_ref"]
            e.check(ref["path"], ref["sha256"])
            refs.append(ref)
            path = (
                Path("outputs/wrench_coverage_interventions_20261025") / f"case{index:02d}" / "collection_result.json"
            )
            e.check(path, closure["source_sha256"][str(path.resolve())])
            collection = e.json(path)
            if contract is None:
                contract = collection["controller_contract"]
            else:
                assert contract == collection["controller_contract"]
            with np.load(ref["path"], allow_pickle=False) as actual:
                np.testing.assert_allclose(actual["controller_site_pose"][anchor], current, rtol=0, atol=1e-9)
                np.testing.assert_allclose(
                    actual["source_twist"][anchor], condition["x_site_twist"][0], rtol=0, atol=1e-9
                )
                packet = dict(
                    command=actual["sent_commands"][-1].copy(),
                    goal_pos=actual["controller_goal_pos"][-1].copy(),
                    goal_ori=actual["controller_goal_ori"][-1].copy(),
                )
                target = np.r_[
                    actual["controller_site_pose"][-1, :3, 3] - current[:3, 3],
                    Rotation.from_matrix(actual["controller_site_pose"][-1, :3, :3] @ current[:3, :3].T).as_rotvec(),
                    0.05 * actual["source_twist"][-1],
                ]
                packets.append(packet)
                targets.append(target)
        scale = (np.array(contract["output_max"])[:3] - np.array(contract["output_min"])[:3]) / (
            np.array(contract["input_max"])[:3] - np.array(contract["input_min"])[:3]
        )
        baseline_delta = packets[0]["goal_pos"] - current[:3, 3]
        baseline_rotation = Rotation.from_matrix(packets[0]["goal_ori"] @ current[:3, :3].T).as_rotvec()
        for axis in range(3):
            minus, plus = 1 + 2 * axis, 2 + 2 * axis
            delta = float(packets[plus]["command"][axis] - packets[minus]["command"][axis])
            assert delta > 0
            selected = []
            for branch in (minus, plus):
                packet = packets[branch]
                other = [k for k in range(7) if k != axis]
                np.testing.assert_array_equal(packet["command"][other], packets[0]["command"][other])
                np.testing.assert_allclose(packet["goal_ori"], packets[0]["goal_ori"], rtol=0, atol=1e-9)
                canonical = baseline_delta + scale * (packet["command"][:3] - packets[0]["command"][:3])
                np.testing.assert_allclose(packet["goal_pos"] - current[:3, 3], canonical, rtol=0, atol=1e-9)
                row = {k: v.copy() for k, v in condition.items()}
                row["action_command"][0] = packet["command"]
                row["action_goal_delta"][0] = np.r_[canonical, baseline_rotation]
                selected.append(row)
            xm, xp = (design(row, parameters)[0] for row in selected)
            bm, bp = (nominal(row, parameters)[0] for row in selected)
            feature = (xp - xm) / delta
            assert np.max(np.abs(feature[:218])) == 0
            x.append(feature)
            y.append((targets[plus] - targets[minus]) / delta)
            base.append((bp - bm) / delta)
            records.append(
                dict(
                    anchor_group=group,
                    axis=axis,
                    actual_normalized_delta=delta,
                    minus_case=group * 7 + minus,
                    plus_case=group * 7 + plus,
                    baseline_ref=refs[0],
                    minus_ref=refs[minus],
                    plus_ref=refs[plus],
                )
            )
    assert len(x) == 63
    return np.array(x), np.array(y), np.array(base), records
