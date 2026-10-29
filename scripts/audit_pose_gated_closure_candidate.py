"""Evaluate candidate trigger coverage on fixed saved generated references only."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.pose_gated_gripper_reference import align_closure
from s2a.terminal_development_policy import case_order


def main():
    root = REPO / "outputs/pose_gated_closure_candidate_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    parameters = dict(execute_steps=8, closing_threshold=0.35, position_tolerance_m=0.02, rotation_tolerance_rad=0.15)
    sources = [reference(Path(__file__)), reference(REPO / "src/s2a/pose_gated_gripper_reference.py")]
    registration = publish(
        root / "registration.json",
        dict(
            parameters=parameters,
            source_refs=sources,
            case_indices=[0, 1, 2],
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            scope="Fixed recorded-input trigger coverage; no physical intervention or task qualification",
        ),
    )
    cases = []
    base = REPO / "outputs/tracking_preview_wam_development_v3_20261027/development/cases"
    for index in (0, 1, 2):
        case = case_order()[index]
        worker_ref = reference(base / case["case_id"] / "worker_result.json")
        worker = read(worker_ref)
        require(worker["complete"] is worker["passed"] is True, "Closed original case")
        chunks_ref = reference(Path(worker["physical_case_dir"]) / "chunks.json")
        chunks = read(chunks_ref)
        require(len(chunks) == 75, "All original queries")
        triggers = []
        for query, chunk in enumerate(chunks):
            require(chunk["step"] == query * 8, "Original query schedule")
            result = align_closure(chunk["anchor_tcp"], chunk["effective_targets_tcp"], chunk["openings"], **parameters)
            if result["selected_index"] is not None:
                triggers.append(
                    dict(
                        query=query,
                        step=chunk["step"],
                        selected_index=result["selected_index"],
                        pose_error=result["first_deferred_closure_pose_error"],
                        previous_prefix_min=float(np.min(chunk["openings"][:8])),
                        candidate_prefix_min=float(result["openings"][:8].min()),
                    )
                )
        cases.append(
            dict(
                case=case, worker_ref=worker_ref, chunks_ref=chunks_ref, triggers=triggers, trigger_count=len(triggers)
            )
        )
    print(
        publish(
            root / "verified.json",
            dict(
                complete=True,
                passed=True,
                registration_ref=registration,
                cases=cases,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                scope="Descriptive fixed-input intervention coverage; no counterfactual physics, responsive WAM or task success evidence",
            ),
        ),
        flush=True,
    )
    print(json.dumps([dict(case=c["case"]["case_id"], triggers=c["trigger_count"]) for c in cases]), flush=True)


if __name__ == "__main__":
    main()
