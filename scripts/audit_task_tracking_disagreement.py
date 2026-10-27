"""170 compare audited same-profile task/contact outcomes despite small pose error."""

import json

import numpy as np
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    root = REPO / "outputs/task_tracking_disagreement_20261025"
    if root.exists():
        raise FileExistsError(root)
    e = Evidence()
    e.read(__file__)
    names = ("pose_velocity_loaded_control", "causal_residual_loaded_control", "qualified_inverse_enumerated")
    profiles = []
    results = []
    for name in names:
        config = e.json(f"configs/{name}_20261025.json")
        profiles.append(config["profiles"][3])
        case = REPO / f"outputs/{name}_20261025/case03"
        verified = e.json(case / "independent_verified.json")
        assert verified["complete"] and verified["mechanical_labels_images_passed"]
        path = next(case.rglob("steps.jsonl"))
        rows = [json.loads(line) for line in e.read(path).decode().splitlines()]
        assert len(rows) == 170
        objects = {}
        for key in rows[0]["objects"]:
            touches = [int(row["step"]) for row in rows if row["objects"][key]["grasp_contact"]]
            objects[key] = dict(
                grasp_contact_periods=len(touches),
                first_contact_step=touches[0] if touches else None,
                last_contact_step=touches[-1] if touches else None,
                max_height_gain_m=max(row["objects"][key]["height_gain_m"] for row in rows),
                final_position=rows[-1]["objects"][key]["position"],
            )
        error = np.asarray([row["source_error"] for row in rows])
        opening_error = np.asarray([row["opening_after"] - row["opening_target"] for row in rows])
        results.append(
            dict(
                name=name,
                outcome=verified["control"]["outcome"],
                tracking=verified["control"]["tracking"]["admitted"],
                performance_qualified=verified["performance_qualified"],
                objects=objects,
                max_source_position_error_m=float(np.linalg.norm(error[:, :3], axis=1).max()),
                opening_rmse=float(np.sqrt(np.mean(opening_error**2))),
            )
        )
    assert profiles[0] == profiles[1] == profiles[2]
    e.recheck()
    publish(
        root / "verified.json",
        dict(
            complete=True,
            profile=profiles[0],
            cases=results,
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
            control_qualified=False,
            limitation="Audited saved logs and contact flags only; no causal geometry diagnosis or proof that a single error component explains task failure",
        ),
    )
    print(json.dumps(results))


if __name__ == "__main__":
    main()
