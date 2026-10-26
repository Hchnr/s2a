"""136 training-only finite-action even response at matched current conditions."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once
from scipy.spatial.transform import Rotation


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/136_training_action_curvature_20261025.md")
    root = Path("outputs/condition_span_forward_20261025")
    c = e.json(root / "closure_verified.json")
    assert c["complete"] and c["numerically_verified"] and c["owner_normal_wait_verified"]
    path = root / "fit_result.json"
    e.check(path, c["source_sha256"][str(path.resolve())])
    fit = e.json(path)
    assert len(fit["contrast_sources"]) == 198
    rows = []
    for r in fit["contrast_sources"]:
        packets = []
        commands = []
        currents = []
        twists = []
        for key in ("baseline_ref", "minus_ref", "plus_ref"):
            ref = r[key]
            e.check(ref["path"], ref["sha256"])
            with np.load(ref["path"], allow_pickle=False) as a:
                p = a["controller_site_pose"][-2:]
                packets.append(
                    np.r_[
                        p[1, :3, 3] - p[0, :3, 3],
                        Rotation.from_matrix(p[1, :3, :3] @ p[0, :3, :3].T).as_rotvec(),
                        0.05 * a["source_twist"][-1],
                    ]
                )
                commands.append(a["sent_commands"][-1].copy())
                currents.append(p[0].copy())
                twists.append(a["source_twist"][-2].copy())
        for i in (1, 2):
            np.testing.assert_allclose(currents[i], currents[0], rtol=0, atol=1e-9)
            np.testing.assert_allclose(twists[i], twists[0], rtol=0, atol=1e-9)
        axis = r["axis"]
        delta = r["actual_normalized_delta"]
        # Clipped interventions may be asymmetric; use the actual interpolation fraction.
        fraction = (commands[0][axis] - commands[1][axis]) / delta
        assert 0 < fraction < 1
        other = [i for i in range(7) if i != axis]
        for i in (1, 2):
            np.testing.assert_array_equal(commands[i][other], commands[0][other])
        even = packets[0] - ((1 - fraction) * packets[1] + fraction * packets[2])
        odd = packets[2] - packets[1]
        metrics = {}
        for i, name in enumerate(("position", "rotation_vector", "linear_velocity", "angular_velocity")):
            scale = 1 if i < 2 else 20
            a = float(np.linalg.norm(even[i * 3 : i * 3 + 3]) * scale)
            b = float(np.linalg.norm(odd[i * 3 : i * 3 + 3]) * scale)
            metrics[name] = dict(interpolation_error=a, plus_minus_difference=b, ratio=a / b if b > 0 else None)
        rows.append(
            dict(
                dataset=r["dataset"],
                anchor_group=r["anchor_group"],
                axis=axis,
                baseline_fraction=float(fraction),
                metrics=metrics,
            )
        )
    e.recheck()
    out = Path("outputs/training_action_curvature_20261025")
    out.mkdir(exist_ok=False)
    write_once(
        out / "result.json",
        dict(
            complete=True,
            training_only=True,
            equations=rows,
            source_sha256=e.hashes,
            new_fits=0,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )
    for name in rows[0]["metrics"]:
        ratios = [r["metrics"][name]["ratio"] for r in rows if r["metrics"][name]["ratio"] is not None]
        print(name, "median interpolation/branch difference", float(np.median(ratios)), "max", max(ratios))


if __name__ == "__main__":
    main()
