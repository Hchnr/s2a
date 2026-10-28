"""Validate the independent planner implementation at the exact failed native state."""

import json
from pathlib import Path

import numpy as np
from coupled_pose_preview_independent import verify_preview

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.dependency_safe_preview_runtime import solve_preview


def main():
    capture_ref = reference(REPO / "outputs/tracking_wam_qp_failure_capture_v2_20261027/result.json")
    capture = read(capture_ref)
    require(capture["complete"] is capture["passed"] is True, "Exact native failure capture")
    contract = read(capture["controller_ref"])["contract"]
    with np.load(checked(capture["inputs_ref"]), allow_pickle=False) as archive:
        inputs = {k: archive[k].copy() for k in archive.files}
    solved = solve_preview(
        inputs["mass"],
        inputs["jacobian"],
        inputs["pose"],
        inputs["twist"],
        inputs["generated_poses"],
        inputs["query_anchor"],
        2,
        inputs["previous_command"],
        contract,
        innovation=inputs["innovation"],
    )
    require(
        solved["preview"]["qp_recovery"]["route"] == "nullspace_after_dependent_blocker",
        "Recovery implementation actually selected",
    )
    proof = verify_preview(
        solved["preview"],
        inputs["pose"],
        inputs["twist"],
        np.asarray(solved["references"]),
        np.asarray(solved["reference_twists"]),
        inputs["previous_command"],
        inputs["mass"],
        inputs["jacobian"],
        contract,
        innovation=inputs["innovation"],
    )
    previous_ref = reference(REPO / "outputs/tracking_wam_recovered_seed_20261027/verified.json")
    previous = read(previous_ref)
    np.testing.assert_array_equal(solved["command"], previous["command"])
    sources = [
        reference(Path(__file__)),
        *[
            reference(REPO / f"src/s2a/{name}.py")
            for name in [
                "dependency_safe_qp",
                "dependency_safe_pose_preview",
                "dependency_safe_preview_executable",
                "dependency_safe_preview_runtime",
            ]
        ],
    ]
    print(
        json.dumps(
            publish(
                REPO / "outputs/dependency_safe_preview_seed_20261027/verified.json",
                dict(
                    complete=True,
                    passed=True,
                    capture_ref=capture_ref,
                    previous_diagnostic_ref=previous_ref,
                    source_refs=sources,
                    qp_recovery=solved["preview"]["qp_recovery"],
                    command=np.asarray(solved["command"]).tolist(),
                    independent_proof=proof,
                    physics_steps=0,
                    world_model_calls=0,
                    fresh_cases=0,
                    scope="New planner implementation at one captured failure state; not physical control or WAM qualification",
                ),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
