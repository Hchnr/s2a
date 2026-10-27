"""158 reproduce fixed controller rotation precision on original157 logs."""

import json
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once
from run_matched_oracle_case import CONTROL, CONTROL_SHA
from scipy.spatial.transform import Rotation

from s2a.artifacts import sha256
from s2a.native_environment import restore_cv2_environment


def independent_matrix(delta):
    angle = np.linalg.norm(delta)
    q = np.r_[np.cos(angle / 2), delta / angle * np.sin(angle / 2)].astype(np.float32)
    q = (q.astype(np.float64) * np.sqrt(2.0 / float(np.dot(q, q)))).astype(np.float32)
    m = np.outer(q, q)
    return np.array(
        [
            [1.0 - float(m[2, 2]) - float(m[3, 3]), float(m[1, 2] - m[3, 0]), float(m[1, 3] + m[2, 0])],
            [float(m[1, 2] + m[3, 0]), 1.0 - float(m[1, 1]) - float(m[3, 3]), float(m[2, 3] - m[1, 0])],
            [float(m[1, 3] - m[2, 0]), float(m[2, 3] + m[1, 0]), 1.0 - float(m[1, 1]) - float(m[2, 2])],
        ]
    )


def main():
    assert sha256(CONTROL) == CONTROL_SHA
    guard = json.loads(CONTROL.read_text())["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from robosuite.utils import transform_utils as transform

    e = Evidence()
    for p in (__file__, transform.__file__, CONTROL, "plans/v0.0.1/158_rotation_goal_precision_incident_20261025.md"):
        e.read(p)
    root = Path("outputs/qualified_inverse_control_20261025")
    exited = e.json(root / "owner_wait/exit.json")
    released = e.json(root / "owner_wait/release.json")
    assert exited["returncode"] == 1 and exited["actual_wait"] and released["reaped"] and not released["signal_sent"]
    case = next((root / "case00").glob("simulation/*/*"))
    rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
    contract = e.json(case / "controller.json")["contract"]
    lo, hi, outlo, outhi = (np.asarray(contract[k]) for k in ("input_min", "input_max", "output_min", "output_max"))
    results = []
    for row in rows:
        packet = row["qualified_forward_input"]
        pose = np.asarray(packet["x_site_pose"])[0]
        scaled = outlo + (np.asarray(row["command"])[:6] - lo) * (outhi - outlo) / (hi - lo)
        actual = np.asarray(row["controller_goal_after"])
        if np.all(scaled[3:] == 0):
            predicted = np.asarray(row["controller_goal_before"])
        else:
            delta = scaled[3:]
            rotation = transform.quat2mat(transform.axisangle2quat(delta))
            np.testing.assert_allclose(independent_matrix(delta), rotation, rtol=0, atol=4*np.finfo(np.float32).eps)
            predicted = rotation @ pose[:3, :3]
        np.testing.assert_allclose(predicted, actual, rtol=0, atol=1e-9)
        exact = Rotation.from_matrix(actual @ pose[:3, :3].T).as_rotvec()
        logged = np.asarray(packet["action_goal_delta"])[0, 3:]
        results.append(
            dict(
                step=row["step"],
                goal_vector_difference=float(np.max(np.abs(exact - logged))),
                matrix_reproduction_error=float(np.max(np.abs(predicted - actual))),
            )
        )
    e.recheck()
    out = Path("outputs/rotation_goal_precision_incident_20261025")
    out.mkdir(exist_ok=False)
    write_once(
        out / "result.json",
        dict(
            complete=True,
            periods=len(rows),
            maximum_goal_vector_difference=max(r["goal_vector_difference"] for r in results),
            maximum_matrix_reproduction_error=max(r["matrix_reproduction_error"] for r in results),
            records=results,
            source_sha256=e.hashes,
            physics_steps=0,
            new_fits=0,
            control_qualified=False,
        ),
    )
    print(
        "periods",
        len(rows),
        "maximum vector difference",
        max(r["goal_vector_difference"] for r in results),
        "actual matrix reproduced",
    )


if __name__ == "__main__":
    main()
