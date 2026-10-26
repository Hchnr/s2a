"""179 measured aperture-change labels aligned to frozen146 equations."""

import numpy as np
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish, sha


def gap(q):
    q = np.asarray(q, dtype=float)
    assert q.ndim == 2 and q.shape[1] == 2 and np.isfinite(q).all()
    return q[:, 0] - q[:, 1]


def main():
    root = REPO / "outputs/arm_finger_response_labels_20261025"
    if root.exists():
        raise FileExistsError(root)
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/179_arm_finger_response_labels_20261025.md")
    source = e.json("outputs/native_arm_response_data_20261025/reconstruction_result.json")
    equations = e.json("outputs/left_table_forward_equations_20261025/result.json")
    fit, heldout, records = [], [], []
    for entry in source["trajectories"]:
        ref, current_ref = entry["source"], entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        e.check(current_ref["path"], current_ref["sha256"])
        with (
            np.load(ref["path"], allow_pickle=False) as actual,
            np.load(current_ref["path"], allow_pickle=False) as current,
        ):
            q = actual["gripper_qpos"].copy()
            np.testing.assert_allclose(q[:-1], current["x_finger_qpos"], atol=1e-12, rtol=0)
            label = np.diff(gap(q))
            assert len(label) == len(current["x_site_pose"])
        (fit if entry["split"] == "fit" else heldout).append(label)
        records.append(dict(native_index=entry["index"], split=entry["split"], rows=len(label)))
    contrasts = []
    for row in equations["contrast_sources"]:
        changes = []
        before = []
        for key in ("minus_ref", "plus_ref"):
            ref = row[key]
            e.check(ref["path"], ref["sha256"])
            with np.load(ref["path"], allow_pickle=False) as actual:
                q = actual["gripper_qpos"][-2:].copy()
                changes.append(float(np.diff(gap(q))[0]))
                before.append(q[0])
        np.testing.assert_allclose(before[0], before[1], atol=1e-12, rtol=0)
        contrasts.append((changes[1] - changes[0]) / row["actual_normalized_delta"])
    fit, heldout, contrasts = np.concatenate(fit), np.concatenate(heldout), np.asarray(contrasts)
    assert fit.shape == (4865,) and heldout.shape == (1315,) and contrasts.shape == (207,)
    e.recheck()
    root.mkdir()
    target_path = root / "labels.npz"
    with target_path.open("xb") as stream:
        np.savez_compressed(stream, fit_target=np.r_[fit, contrasts], holdout_target=heldout)
    publish(
        root / "result.json",
        dict(
            complete=True,
            label_units="metres: delta(q_left-q_right)",
            fit_natural_rows=4865,
            contrast_equations=207,
            holdout_rows=1315,
            records=records,
            labels_ref=dict(path=str(target_path), sha256=sha(target_path)),
            source_sha256=e.hashes,
            fits=0,
            physics_steps=0,
            control_qualified=False,
            independently_verified=False,
        ),
    )
    print("179 labels prepared;5072fit/1315holdout;no fit or physics;independent verification pending")


if __name__ == "__main__":
    main()
