"""Compare two mathematically equivalent reference multiplication orders."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.coupled_pose_preview_runtime import solve_preview
from s2a.geometry import inverse


def main():
    entry = read(reference(REPO / "outputs/tracking_correct_reference_20261027/prepared.json"))["profiles"][6]
    old_rows = [json.loads(line) for line in checked(entry["baseline_rows_ref"]).read_text().splitlines()]
    original_chunks = read(reference(Path(entry["baseline_rows_ref"]["path"]).parent / "chunks.json"))
    root = REPO / "outputs/tracking_correct_reference_20261027/case06_candidate"
    row = read(reference(root / "steps.json"))[0]
    chunk = read(reference(root / "chunks.json"))[0]
    controller = read(entry["controller_ref"])
    mapping = np.asarray(controller["source_from_site"])
    tool = np.asarray(read(entry["tool_ref"])["source_from_tcp"])
    targets, anchor = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["anchor_tcp"])
    np.testing.assert_array_equal(targets, original_chunks[0]["effective_targets_tcp"])
    d = row["dynamics"]
    outcomes = {}
    for label, sites, site_anchor in [
        ("port", targets @ inverse(inverse(mapping) @ tool), anchor @ inverse(inverse(mapping) @ tool)),
        ("original_order", targets @ inverse(tool) @ mapping, anchor @ inverse(tool) @ mapping),
    ]:
        solved = solve_preview(
            np.asarray(d["mass"]),
            np.asarray(d["jacobian"]),
            np.asarray(row["site_pose"]),
            np.asarray(row["site_twist"]),
            sites,
            site_anchor,
            0,
            np.zeros(6),
            controller["contract"],
            innovation=np.zeros(12),
        )
        command = np.asarray(solved["command"])
        outcomes[label] = dict(
            command=command.tolist(),
            maximum_original_difference=float(np.max(abs(command - np.asarray(old_rows[0]["command"])[:6]))),
            original_exact=bool(np.array_equal(command, old_rows[0]["command"][:6])),
        )
    require(outcomes["port"]["maximum_original_difference"] > 0, "Reproduce numerical difference")
    print(
        json.dumps(
            publish(
                REPO / "outputs/tracking_reference_roundoff_native_20261027/diagnostic.json",
                dict(
                    complete=True,
                    passed=True,
                    source_ref=reference(Path(__file__)),
                    baseline_rows_ref=entry["baseline_rows_ref"],
                    outcomes=outcomes,
                    physics_steps=0,
                    world_model_calls=0,
                    fresh_cases=0,
                    scope="First command only; no evidence of complete-case recovery",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
