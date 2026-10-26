"""184 recover local validation slopes from raw states, not observed finger arrays."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from verify_arm_finger_response_labels import aperture

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    out = REPO / "outputs/arm_finger_local_independent_20261025"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    e.read(__file__)
    e.read("scripts/verify_arm_finger_response_labels.py")
    results = e.json("outputs/arm_finger_local_response_20261025/result.json")
    config = e.json("configs/left_table_forward_20261025.json")
    closed = e.json(config["paired_closure_ref"]["path"])
    summary = e.json(closed["summary_ref"]["path"])
    for row in results["results"]:
        group = row["group"]
        slopes = []
        for axis in range(3):
            labels, commands = [], []
            for branch in (1 + 2 * axis, 2 + 2 * axis):
                ref = summary["cases"][7 * group + branch]["arrays_ref"]
                e.check(ref["path"], ref["sha256"])
                xml = e.read(Path(ref["path"]).parent / "model.xml").decode()
                with np.load(ref["path"], allow_pickle=False) as trace:
                    qgap = aperture(trace["sim_states"][-2:], xml)
                    labels.append(qgap[1] - qgap[0])
                    commands.append(trace["sent_commands"][-1, axis])
            slopes.append((labels[1] - labels[0]) / (commands[1] - commands[0]))
        np.testing.assert_allclose(slopes, row["measured"], atol=1e-12, rtol=0)
        error = float(np.linalg.norm(np.asarray(row["predicted"]) - slopes))
        baseline = float(np.linalg.norm(slopes))
        assert (error <= baseline) == row["passed"]
        np.testing.assert_allclose(
            [error, baseline], [row["model_error"], row["zero_coupling_error"]], atol=1e-12, rtol=0
        )
    e.recheck()
    publish(
        out / "verified.json",
        dict(
            complete=True,
            independent_raw_state_slopes_verified=True,
            passed=results["passed"],
            local_qualified=results["local_qualified"],
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
            control_qualified=False,
        ),
    )
    print("184 raw-state local slopes confirm2/6;model not eligible for control")


if __name__ == "__main__":
    main()
