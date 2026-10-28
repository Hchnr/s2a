"""Describe generated future closure versus actual eight-period reference consumption."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.terminal_development_policy import case_order


def main():
    root = REPO / "outputs/tracking_preview_wam_development_v3_20261027/development/cases"
    cases = []
    for index in (0, 1, 2):
        case = case_order()[index]
        worker_ref = reference(root / case["case_id"] / "worker_result.json")
        worker = read(worker_ref)
        require(worker["complete"] is worker["passed"] is True, "Closed case only")
        folder = Path(worker["physical_case_dir"])
        chunks_ref = reference(folder / "chunks.json")
        chunks = read(chunks_ref)
        require(len(chunks) == 75, "All75 references")
        entries = []
        for j, chunk in enumerate(chunks):
            require(chunk["step"] == j * 8, "Actual eight-period query schedule")
            opening = np.asarray(chunk["openings"])
            poses = np.asarray(chunk["effective_targets_tcp"])
            anchor = np.asarray(chunk["anchor_tcp"])
            require(opening.shape == (40,) and poses.shape == (40, 4, 4), "Complete generated40")
            close = np.flatnonzero(opening < 0.35)
            entries.append(
                dict(
                    query=j,
                    executed_minimum=float(opening[:8].min()),
                    future_minimum=float(opening[8:].min()),
                    full_minimum=float(opening.min()),
                    deferred_closure=bool(opening[:8].min() >= 0.35 and opening[8:].min() < 0.35),
                    first_closure_index=None if not len(close) else int(close[0]),
                    first_generated_closure_distance_from_current_tcp_m=None
                    if not len(close)
                    else float(np.linalg.norm(poses[close[0], :3, 3] - anchor[:3, 3])),
                )
            )
        cases.append(
            dict(
                case=case,
                worker_ref=worker_ref,
                chunks_ref=chunks_ref,
                deferred_closure_queries=sum(e["deferred_closure"] for e in entries),
                queries_with_generated_closure=sum(e["first_closure_index"] is not None for e in entries),
                minimum_executed_opening=min(e["executed_minimum"] for e in entries),
                minimum_generated_opening=min(e["full_minimum"] for e in entries),
                queries=entries,
            )
        )
    print(
        json.dumps(
            publish(
                REPO / "outputs/tracking_wam_interactions_20261027/reference_phase_first_three.json",
                dict(
                    complete=True,
                    passed=True,
                    source_ref=reference(Path(__file__)),
                    cases=cases,
                    threshold_point35_descriptive_not_qualification=True,
                    physics_steps=0,
                    world_model_calls=0,
                    fresh_cases=0,
                    scope="Saved generated references versus executed prefix; not causal evidence or permission to change online schedule",
                ),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
