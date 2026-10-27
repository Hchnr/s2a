"""Check reusable preview against the frozen local hold sequences, not physical outcomes."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from scipy.spatial.transform import Rotation

from s2a.coupled_pose_preview import plan_pose
from s2a.durable_stage_supervisor import publish


def main():
    output = Path("outputs/coupled_pose_preview_extraction_20261026/verified.json")
    assert not output.exists()
    e = Evidence()
    for p in (
        __file__,
        "src/s2a/coupled_pose_preview.py",
        "src/s2a/strict_convex_qp.py",
        "src/s2a/current_coupled_response.py",
        "src/s2a/geometry.py",
        "src/s2a/osc.py",
    ):
        e.read(p)
    model_proof = e.json("outputs/full_coupled_sqp_hold_model_20261026/verified.json")
    sequences = e.json("outputs/full_coupled_sqp_hold_preview_20261026/verified.json")
    assert model_proof["complete"] and sequences["complete"]
    for proof in (model_proof, sequences):
        for p, digest in proof["source_sha256"].items():
            e.check(p, digest)
    checks = []
    for record in model_proof["results"]:
        root = Path("outputs/full_coupled_sqp_control_20261026") / f"case{record['case']:02d}"
        case = next(root.glob("simulation/*/*"))
        contract = e.json(case / "controller.json")["contract"]
        for horizon in (1, 4):
            saved = next(x for x in sequences["results"] if x["case"] == record["case"] and x["horizon"] == horizon)
            previous = np.asarray(record["previous_command"])
            maximum = 0.0
            for state, expected in zip(saved["states"][:-1], saved["commands"], strict=True):
                state = np.asarray(state)
                measured = np.eye(4)
                measured[:3, 3] = state[:3]
                measured[:3, :3] = Rotation.from_rotvec(state[3:6]).as_matrix()
                result = plan_pose(
                    measured,
                    state[6:],
                    np.repeat(np.eye(4)[None], horizon, axis=0),
                    np.zeros((horizon, 6)),
                    previous,
                    record["frozen_mass"],
                    record["frozen_jacobian"],
                    contract,
                    innovation=np.zeros(12),
                )
                maximum = max(maximum, float(np.max(np.abs(result["command"] - expected))))
                np.testing.assert_allclose(result["command"], expected, atol=1e-8, rtol=0)
                previous = np.asarray(expected)
            checks.append(dict(case=record["case"], horizon=horizon, periods=200, max_command_difference=maximum))
            print(checks[-1], flush=True)
    e.recheck()
    publish(
        output,
        dict(
            complete=True,
            scope="reusable planner matches frozen local model sequences; no physical feedback",
            checks=checks,
            source_sha256=e.hashes,
            seed=42,
            new_physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
