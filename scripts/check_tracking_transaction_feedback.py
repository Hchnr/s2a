"""Check active-route transaction parity against independently verified feedback."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require


def main():
    old_verified_ref = reference(REPO / "outputs/tracking_margin_feedback_verified_20261027/verified.json")
    require(
        read(old_verified_ref)["complete"] is read(old_verified_ref)["passed"] is True, "Verified original feedback"
    )
    root = REPO / "outputs/tracking_transaction_feedback_20261027/case00_tracking"
    result_ref = reference(root / "result.json")
    result = read(result_ref)
    require(
        result["complete"] is result["passed"] is True
        and result["prefix_steps"] == 72
        and result["branch_steps"] == 56,
        "Complete prefix and branch",
    )
    rows_ref = reference(root / "steps.json")
    old_rows_ref = reference(REPO / "outputs/tracking_margin_feedback_branches_20261027/case00_tracking/steps.json")
    rows, old = read(rows_ref), read(old_rows_ref)
    require(len(rows) == len(old) == 56, "Full branch")
    max_state = 0.0
    active = 0
    for a, b in zip(rows, old, strict=True):
        for key in [
            "command",
            "sim_state",
            "tcp_error",
            "controller_goal_pos",
            "controller_goal_ori",
            "limited_history_after",
        ]:
            np.testing.assert_allclose(a[key], b[key], atol=1e-8, rtol=0)
        require(
            a["success"] == b["success"] and a["refinement"]["route"] == b["refinement"]["route"],
            "Task and route identity",
        )
        max_state = max(max_state, float(np.max(abs(np.asarray(a["sim_state"]) - b["sim_state"]))))
        active += a["refinement"]["optimization"] is not None
    require(active > 0, "Actual optimization route exercised")
    print(
        json.dumps(
            publish(
                root / "transaction_parity.json",
                dict(
                    complete=True,
                    passed=True,
                    result_ref=result_ref,
                    rows_ref=rows_ref,
                    old_rows_ref=old_rows_ref,
                    old_verified_ref=old_verified_ref,
                    source_ref=reference(Path(__file__)),
                    active_periods=active,
                    maximum_state_difference=max_state,
                    physics_steps=0,
                    world_model_calls=0,
                    fresh_cases=0,
                    scope="Actual optimization-route dispatch/history parity on fixed56-period replay; not task or WAM qualification",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
