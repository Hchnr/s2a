"""Verify the full seed planner at the captured failure using guarded QP recovery."""

import json
from pathlib import Path

import numpy as np
from coupled_pose_preview_independent import verify_preview

from s2a import coupled_pose_preview
from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.coupled_pose_preview_runtime import solve_preview
from s2a.dependency_safe_qp import solve_qp


def main():
    capture_ref = reference(REPO / "outputs/tracking_wam_qp_failure_capture_v2_20261027/result.json")
    capture = read(capture_ref)
    require(capture["passed"], "Exact captured prefix")
    contract = read(capture["controller_ref"])["contract"]
    with np.load(checked(capture["inputs_ref"]), allow_pickle=False) as archive:
        inputs = {k: archive[k].copy() for k in archive.files}
    routes = []
    original = coupled_pose_preview.solve_qp

    def guarded(*args):
        result = solve_qp(*args)
        routes.append(result.get("solver_route", "original_strict"))
        return result

    coupled_pose_preview.solve_qp = guarded
    try:
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
    finally:
        coupled_pose_preview.solve_qp = original
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
    require(routes.count("nullspace_after_dependent_blocker") > 0, "Actual guarded recovery used")
    print(
        json.dumps(
            publish(
                REPO / "outputs/tracking_wam_recovered_seed_20261027/verified.json",
                dict(
                    complete=True,
                    passed=True,
                    capture_ref=capture_ref,
                    source_refs=[reference(Path(__file__)), reference(REPO / "src/s2a/dependency_safe_qp.py")],
                    routes=routes,
                    command=np.asarray(solved["command"]).tolist(),
                    independent_proof=proof,
                    physics_steps=0,
                    world_model_calls=0,
                    fresh_cases=0,
                    scope="Full recovered seed and independent original preview gates at one captured state; no action dispatched, physical qualification or task success",
                ),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
