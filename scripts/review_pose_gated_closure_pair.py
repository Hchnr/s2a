"""Recover verifiable closed-step evidence without rewriting a failed summary."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require


def main():
    root = REPO / "outputs/pose_gated_closure_pair_20261027"
    execution = REPO / "outputs/pose_gated_closure_pair_execution_20261027"
    owner_ref = reference(execution / "result.json")
    require(read(owner_ref)["complete"] is False, "Preserve failed owner status")
    exit_ref = reference(execution / "capture/exit.json")
    release_ref = reference(execution / "capture/ownership/release.json")
    exit_record, release = read(exit_ref), read(release_ref)
    require(
        exit_record["actual_wait"] is release["reaped"] is True
        and exit_record["returncode"] == release["returncode"] == 1,
        "Failed process actually waited and reaped",
    )
    log_ref = reference(execution / "capture/command.log")
    require(
        "TypeError: Object of type bool_ is not JSON serializable" in checked(log_ref).read_text(),
        "Actual summary serialization failure",
    )
    registration_ref = reference(root / "registration.json")
    reg = read(registration_ref)
    original = [json.loads(x) for x in checked(reg["rows_ref"]).read_text().splitlines()]
    records = []
    arrays = {}
    for branch in ("baseline", "candidate"):
        rows_ref = reference(root / branch / "steps.json")
        rows = read(rows_ref)
        require(len(rows) == 600 and [r["step"] for r in rows] == list(range(600)), "Complete ordered branch")
        command = np.asarray([r["command"] for r in rows])
        source = np.asarray([r["command"] for r in original])
        require(command.shape == (600, 7) and np.isfinite(command).all(), "All finite commands")
        np.testing.assert_array_equal(command[:, :6], source[:, :6])
        opening = np.asarray([r["opening"] for r in rows])
        require(
            np.isfinite(opening).all() and np.all(abs(command[:, 6]) <= 1), "Finite opening and feasible motor command"
        )
        success = [r["success"] for r in rows]
        grasp = [r["grasp"] for r in rows]
        require(all(type(x) is bool for x in success + grasp), "Actual logged boolean predicates")
        if branch == "baseline":
            np.testing.assert_array_equal(command, source)
            np.testing.assert_allclose(opening, [r["opening_after"] for r in original], atol=1e-12, rtol=0)
        changed = np.flatnonzero(abs(command[:, 6] - source[:, 6]) > 1e-12)
        records.append(
            dict(
                branch=branch,
                steps_ref=rows_ref,
                recorded_periods=600,
                success_steps=sum(success),
                terminal=success[-1],
                stable=all(success[-40:]),
                grasp_steps=sum(grasp),
                changed_gripper_commands=len(changed),
                first_changed_step=None if not len(changed) else int(changed[0]),
                opening_min=float(opening.min()),
                opening_max=float(opening.max()),
            )
        )
        arrays[branch] = opening
    print(
        publish(
            REPO / "outputs/pose_gated_closure_pair_review_20261027/verified.json",
            dict(
                complete=True,
                passed=True,
                source_ref=reference(Path(__file__)),
                failed_owner_ref=owner_ref,
                failed_summary_ref=reference(root / "result.json"),
                exit_ref=exit_ref,
                release_ref=release_ref,
                log_ref=log_ref,
                registration_ref=registration_ref,
                branches=records,
                maximum_opening_difference=float(np.max(abs(arrays["candidate"] - arrays["baseline"]))),
                recorded_physics_periods=1200,
                new_physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                original_owner_passed=False,
                scope="Closed producer step-log review only; failed summary retained; no independent physics reroll or WAM/fresh qualification",
            ),
        ),
        flush=True,
    )
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
