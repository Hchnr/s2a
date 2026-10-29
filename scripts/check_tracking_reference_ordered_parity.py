"""Port admission: compare a full reconstructed baseline with the qualified run."""

import argparse
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=range(18), required=True)
    args = parser.parse_args()
    prepared = read(reference(REPO / "outputs/tracking_correct_reference_20261027/prepared.json"))
    entry = prepared["profiles"][args.case_index]
    root = REPO / f"outputs/tracking_correct_reference_ordered_20261027/case{args.case_index:02d}_baseline"
    result_ref = reference(root / "result.json")
    result = read(result_ref)
    require(result["complete"] is result["passed"] is True, "Complete baseline execution")
    old = read(entry["baseline_collection_ref"])
    differences = {}
    with (
        np.load(checked(old["arrays_ref"]), allow_pickle=False) as a,
        np.load(checked(result["arrays_ref"]), allow_pickle=False) as b,
    ):
        for key in (
            "sim_states",
            "source_pose",
            "controller_site_pose",
            "opening",
            "gripper_current_action_offsets",
            "gripper_current_action_flat",
        ):
            require(a[key].shape == b[key].shape, f"{key} shape")
            np.testing.assert_allclose(a[key], b[key], atol=1e-8, rtol=0)
            differences[key] = float(np.max(abs(a[key] - b[key])))
    rows = read(reference(root / "steps.json"))
    old_rows = [json.loads(line) for line in checked(entry["baseline_rows_ref"]).read_text().splitlines()]
    require(len(rows) == len(old_rows) == entry["profile"]["physics_steps"], "Full original budget")
    for a, b in zip(old_rows, rows, strict=True):
        np.testing.assert_allclose(a["command"], b["command"], atol=1e-8, rtol=0)
        np.testing.assert_allclose(a["controller_goal_after"], b["controller_goal_ori"], atol=1e-8, rtol=0)
        np.testing.assert_allclose(a["controller_goal_position_after"], b["controller_goal_pos"], atol=1e-8, rtol=0)
        require(a["success"] == b["success"], "Native task predicate")
    require(result["performance_qualified"], "Original case outcome preserved")
    print(
        json.dumps(
            publish(
                root / "port_parity.json",
                dict(
                    complete=True,
                    passed=True,
                    result_ref=result_ref,
                    original_collection_ref=entry["baseline_collection_ref"],
                    source_ref=reference(Path(__file__)),
                    maximum_endpoint_differences=differences,
                    physics_steps=0,
                    world_model_calls=0,
                    fresh_cases=0,
                    scope="full baseline port parity for this original profile; not new-candidate or18-case qualification",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
