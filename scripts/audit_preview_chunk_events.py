"""Locate generated gripper events inside/outside actually executed chunk prefixes."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require


def main():
    root = REPO / "outputs/serialized_preview_wam_development_20261026/development/cases"
    cases = []
    for worker_path in sorted(root.glob("*/worker_result.json")):
        worker = read(reference(worker_path))
        if not worker["complete"]:
            continue
        folder = next(worker_path.parent.glob("simulation/*/*/chunks.json")).parent
        chunks_ref, rows_ref = reference(folder / "chunks.json"), reference(folder / "steps.jsonl")
        chunks = read(chunks_ref)
        rows = [json.loads(line) for line in Path(rows_ref["path"]).read_text().splitlines()]
        require(len(chunks) == 75 and len(rows) == 600, "Fixed complete75/600 cases required")
        events = []
        for i, chunk in enumerate(chunks):
            require(chunk["step"] == 8 * i, "Actual8 step query schedule differs")
            opening = np.clip(np.asarray(chunk["effective128"])[:, 9], 0, 1)
            require(opening.shape == (40,) and np.isfinite(opening).all(), "Full generated horizon required")
            np.testing.assert_array_equal(opening[:8], [r["opening_target"] for r in rows[8 * i : 8 * i + 8]])
            closed, opened = np.flatnonzero(opening < 0.2), np.flatnonzero(opening > 0.8)
            events.append(
                dict(
                    query_index=i,
                    physical_step=chunk["step"],
                    current_opening=chunk["anchor_opening"],
                    first_close_index=int(closed[0]) if len(closed) else None,
                    first_open_index=int(opened[0]) if len(opened) else None,
                    prefix_all_above_point8=bool(np.all(opening[:8] > 0.8)),
                )
            )
        closes = [e for e in events if e["first_close_index"] is not None]
        releases = [e for e in events if e["current_opening"] < 0.2 and e["first_open_index"] is not None]
        cases.append(
            dict(
                case=worker["case"],
                events=events,
                generated_close_chunks=len(closes),
                close_only_after_executed_prefix=sum(e["first_close_index"] >= 8 for e in closes),
                all_open_prefix_chunks=sum(e["prefix_all_above_point8"] for e in events),
                generated_release_from_closed_chunks=len(releases),
                release_only_after_executed_prefix=sum(e["first_open_index"] >= 8 for e in releases),
                input_refs=[reference(worker_path), chunks_ref, rows_ref],
            )
        )
    require(len(cases) == 8, "All8 completed cases required")
    result = dict(
        complete=True,
        passed=True,
        cases=cases,
        source_ref=reference(Path(__file__)),
        actual_prefix_targets_exactly_matched=True,
        scope="descriptive event timing; not counterfactual task success",
        thresholds=dict(close_below=0.2, open_above=0.8),
        new_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    print(json.dumps(publish(REPO / "outputs/preview_chunk_events_20261027/verified.json", result)))


if __name__ == "__main__":
    main()
