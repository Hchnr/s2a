"""186 raw-state finite-branch aperture curvature, without fitting."""

import numpy as np
from audit_wam_position import Evidence
from verify_arm_finger_response_labels import aperture

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    out = REPO / "outputs/finger_action_curvature_20261025"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    e.read(__file__)
    e.read("scripts/verify_arm_finger_response_labels.py")
    e.read("plans/v0.0.1/186_finger_action_curvature_20261025.md")
    config = e.json("configs/left_table_forward_20261025.json")
    closure = e.json(config["paired_closure_ref"]["path"])
    assert closure["complete"] and closure["data_qualified"]
    summary = e.json(closure["summary_ref"]["path"])
    results = []
    for group in range(6):
        values, commands, starting = [], [], []
        for branch in range(7):
            ref = summary["cases"][7 * group + branch]["arrays_ref"]
            e.check(ref["path"], ref["sha256"])
            from pathlib import Path

            xml = e.read(Path(ref["path"]).parent / "model.xml").decode()
            with np.load(ref["path"], allow_pickle=False) as trace:
                gap = aperture(trace["sim_states"][-2:], xml)
                values.append(float(gap[1] - gap[0]))
                starting.append(float(gap[0]))
                commands.append(trace["sent_commands"][-1].copy())
        np.testing.assert_allclose(starting, starting[0], atol=1e-12, rtol=0)
        axes = []
        for axis in range(3):
            minus, plus = 1 + 2 * axis, 2 + 2 * axis
            left = commands[0][axis] - commands[minus][axis]
            right = commands[plus][axis] - commands[0][axis]
            assert left > 0 and right > 0
            interpolation = (right * values[minus] + left * values[plus]) / (left + right)
            mismatch = values[0] - interpolation
            central_amplitude = abs(values[plus] - values[minus]) / 2
            axes.append(
                dict(
                    axis=axis,
                    minus_delta_gap_m=values[minus],
                    baseline_delta_gap_m=values[0],
                    plus_delta_gap_m=values[plus],
                    left_command_step=float(left),
                    right_command_step=float(right),
                    left_slope_m=float((values[0] - values[minus]) / left),
                    right_slope_m=float((values[plus] - values[0]) / right),
                    affine_interpolation_error_m=float(mismatch),
                    central_amplitude_m=float(central_amplitude),
                    nonlinearity_over_central=float(abs(mismatch) / max(central_amplitude, 1e-15)),
                )
            )
        results.append(dict(group=group, used_for_fit=group in config["training_anchor_groups"], axes=axes))
    e.recheck()
    publish(
        out / "verified.json",
        dict(complete=True, results=results, source_sha256=e.hashes, fits=0, physics_steps=0, control_qualified=False),
    )
    for row in results:
        print(
            row["group"],
            [
                (
                    a["axis"],
                    round(a["affine_interpolation_error_m"] * 1000, 6),
                    round(a["nonlinearity_over_central"], 3),
                )
                for a in row["axes"]
            ],
        )


if __name__ == "__main__":
    main()
