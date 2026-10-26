"""090 independently assemble selected finite command differences from raw probe endpoints."""

from pathlib import Path

import numpy as np
from arm_history_independent import INPUTS, augment, feature_matrix, nominal_prediction
from scipy.spatial.transform import Rotation


def equations(config, e, parameters):
    closed = e.json(config["paired_closure_ref"]["path"])
    ref = closed["summary_ref"]
    e.check(ref["path"], ref["sha256"])
    summary = e.json(ref["path"])
    registry = e.json("configs/paired_arm_response_20261025.json")
    original = e.json(config["data_result_ref"]["path"])
    assert config["training_anchor_groups"] == [0, 3, 5]
    features = []
    labels = []
    nominals = []
    identities = []
    for group in (0, 3, 5):
        native = registry["profiles"][group // 2]["trajectory_ref"]
        source = next(item for item in original["trajectories"] if item["source"]["sha256"] == native["sha256"])
        assert source["split"] == "fit"
        k = registry["trials"][7 * group]["anchor_cycle"]
        ref = source["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as data:
            full = augment({key: data[key].copy() for key in data.files}, parameters)
            condition = {name: full[name][k : k + 1].copy() for name in INPUTS}
        current = condition["x_site_pose"][0]

        def load(case):
            row = summary["cases"][case]
            ref = row["arrays_ref"]
            e.check(ref["path"], ref["sha256"])
            path = Path("outputs/paired_arm_response_20261025") / f"case{case:02d}" / "collection_result.json"
            e.check(path, closed["source_sha256"][str(path.resolve())])
            meta = e.json(path)
            with np.load(ref["path"], allow_pickle=False) as data:
                np.testing.assert_allclose(data["controller_site_pose"][k], current, rtol=0, atol=1e-9)
                return dict(
                    command=data["sent_commands"][-1].copy(),
                    goal=data["controller_goal_pos"][-1].copy(),
                    rotation_goal=data["controller_goal_ori"][-1].copy(),
                    pose=data["controller_site_pose"][-1].copy(),
                    twist=data["source_twist"][-1].copy(),
                    contract=meta["controller_contract"],
                )

        baseline = load(7 * group)
        rot = Rotation.from_matrix(baseline["rotation_goal"] @ current[:3, :3].T).as_rotvec()
        for axis in (0, 1, 2):
            left_index = 7 * group + 2 * axis + 1
            right_index = left_index + 1
            left, right = load(left_index), load(right_index)
            span = float(right["command"][axis] - left["command"][axis])
            assert span > 0
            rows = []
            for packet in (left, right):
                other = [a for a in range(7) if a != axis]
                np.testing.assert_array_equal(packet["command"][other], baseline["command"][other])
                np.testing.assert_allclose(packet["rotation_goal"], baseline["rotation_goal"], rtol=0, atol=1e-9)
                row = {key: value.copy() for key, value in condition.items()}
                row["action_command"][0] = packet["command"]
                row["action_goal_delta"][0] = np.r_[packet["goal"] - current[:3, 3], rot]
                rows.append(row)
            xm, xp = (feature_matrix(row, parameters)[0] for row in rows)
            nm, npred = (nominal_prediction(row, parameters)[0] for row in rows)
            rotation_left = Rotation.from_matrix(left["pose"][:3, :3] @ current[:3, :3].T).as_rotvec()
            rotation_right = Rotation.from_matrix(right["pose"][:3, :3] @ current[:3, :3].T).as_rotvec()
            label = (
                np.r_[
                    right["pose"][:3, 3] - left["pose"][:3, 3],
                    rotation_right - rotation_left,
                    0.05 * (right["twist"] - left["twist"]),
                ]
                / span
            )
            features.append((xp - xm) / span)
            labels.append(label)
            nominals.append((npred - nm) / span)
            identities.append((group, axis, left_index, right_index))
    assert len(features) == 9
    return np.array(features), np.array(labels), np.array(nominals), identities
