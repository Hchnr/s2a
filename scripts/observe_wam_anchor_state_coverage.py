"""Offline current-state distance to registered training anchors, without labels."""

import json
from pathlib import Path

import numpy as np
from analyze_wam_release_horizon import reference
from scipy.spatial.transform import Rotation

ROOT = Path("outputs/terminal_development_20261025")
PREPARED = Path(
    "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/preparation/prepared.json"
)


def main():
    prepared_ref = reference(PREPARED)
    prepared = json.loads(PREPARED.read_text())
    anchors = {task: [] for task in range(3)}
    consumed = [prepared_ref]
    for anchor in prepared["training_anchors"]:
        path = Path(anchor["trajectory_ref"]["path"])
        if reference(path) != anchor["trajectory_ref"]:
            raise ValueError("Training trajectory changed")
        frame = anchor["anchor_frame"]
        with np.load(path, allow_pickle=False) as data:
            pose = data["source_pose"][frame].copy()
            opening = float(data["opening"][frame])
        anchors[anchor["source_identity"]["task_index"]].append((pose, opening))
        consumed.append(reference(path))
    if any(len(a) != 50 for a in anchors.values()):
        raise ValueError("Requires original150 registered anchors")
    cases = []
    for path in sorted((ROOT / "development/cases").glob("*/simulation/*/*/steps.jsonl")):
        before = reference(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 600:
            raise ValueError("Requires complete case")
        task = int(path.parents[3].name[4])
        poses = np.array([a[0] for a in anchors[task]])
        openings = np.array([a[1] for a in anchors[task]])
        records = []
        for row in rows[::8]:
            measured = np.asarray(row["measured_source"])
            distances = np.linalg.norm(poses[:, :3, 3] - measured[:3, 3], axis=1)
            index = int(distances.argmin())
            angle = float(Rotation.from_matrix(poses[index, :3, :3].T @ measured[:3, :3]).magnitude())
            records.append(
                dict(
                    step=row["step"],
                    nearest_position_m=float(distances[index]),
                    rotation_at_same_neighbor_rad=angle,
                    opening_at_same_neighbor_abs=float(abs(openings[index] - row["opening_before"])),
                )
            )
        if reference(path) != before:
            raise RuntimeError("Completed log changed")
        values = [r["nearest_position_m"] for r in records]
        cases.append(
            dict(
                case=path.parents[3].name,
                steps_ref=before,
                queries=records,
                nearest_position_quantiles_m=np.quantile(values, [0, 0.5, 0.9, 1]).tolist(),
            )
        )
    if len(cases) != 12 or any(reference(Path(ref["path"])) != ref for ref in consumed):
        raise RuntimeError("Cohort incomplete or inputs changed")
    result = dict(
        scope="Current robot state coverage only; nearest position ignores scene/objects, not a calibrated OOD or causal test",
        script_ref=reference(Path(__file__)),
        input_refs=consumed,
        cases=cases,
        new_model_calls=0,
        new_physics_steps=0,
        future_label_rows_consumed=0,
        fresh_cases=0,
    )
    with (ROOT / "anchor_state_coverage_description.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    for case in cases:
        print(case["case"], case["nearest_position_quantiles_m"])


if __name__ == "__main__":
    main()
