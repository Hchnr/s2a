"""Replay all saved pre-failure QPs algebraically; no simulation or model calls."""

import json
from pathlib import Path

import numpy as np

from s2a import coupled_pose_preview
from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, reference
from s2a.nullspace_convex_qp import solve_qp


def main():
    folder = (
        REPO
        / "outputs/serialized_preview_wam_development_20261026/development/cases/task2_init0_s42_raw/simulation/8400_masked_position_hold/task2_init0_s42"
    )
    rows_path, contract_path = folder / "steps.jsonl", folder / "controller.json"
    contract = json.loads(contract_path.read_text())["contract"]
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    checks = []
    current = None

    def checked(h, g, a, b, initial):
        solved = solve_qp(h, g, a, b, initial)
        x, dual = solved["solution"], solved["multipliers"]
        active = solved["active"]
        stationarity = float(np.max(np.abs(h @ x + g + a[active].T @ dual)))
        slack = b - a @ x
        complementarity = float(np.max(np.abs(dual * slack[active]))) if len(active) else 0.0
        assert slack.min() >= -1e-9 and stationarity <= 1e-8 and complementarity <= 1e-8
        assert np.all(dual >= 0)
        old = np.asarray(current["pose_preview_inverse"]["preview"]["planned_commands"]).ravel()
        delta = x - old
        # Difference form avoids cancellation of a shared large constant.
        objective_delta = float(delta @ (h @ old + g) + 0.5 * delta @ h @ delta)
        checks.append(
            dict(
                step=current["step"],
                stationarity=stationarity,
                primal_violation=max(0.0, float(-slack.min())),
                complementarity=complementarity,
                objective_delta=objective_delta,
                command_max_difference=float(np.max(np.abs(delta))),
                iterations=solved["iterations"],
            )
        )
        return solved

    original = coupled_pose_preview.solve_qp
    coupled_pose_preview.solve_qp = checked
    try:
        for current in rows:
            q, dynamics, plan = (
                current["qualified_forward_input"],
                current["coupled_dynamics"],
                current["pose_preview_inverse"],
            )
            coupled_pose_preview.plan_pose(
                np.asarray(q["x_site_pose"])[0],
                np.asarray(q["x_site_twist"])[0],
                np.asarray(plan["references"]),
                np.asarray(plan["reference_twists"]),
                np.asarray(q["x_previous_command"])[0, :6],
                dynamics["mass"],
                dynamics["jacobian"],
                contract,
                innovation=current["coupled_response_history"]["innovation_before"],
            )
    finally:
        coupled_pose_preview.solve_qp = original
    assert len(checks) == 533
    result = dict(
        passed=True,
        complete=True,
        cases=checks,
        qp_count=len(checks),
        maximum_command_difference=max(c["command_max_difference"] for c in checks),
        maximum_objective_increase=max(c["objective_delta"] for c in checks),
        maximum_stationarity=max(c["stationarity"] for c in checks),
        input_refs=[reference(rows_path), reference(contract_path)],
        source_refs=[reference(Path(__file__)), reference(REPO / "src/s2a/nullspace_convex_qp.py")],
        scope="533 saved QPs only; not new physical control qualification",
        new_model_calls=0,
        physics_steps=0,
    )
    print(json.dumps(publish(REPO / "outputs/nullspace_preview_qp_audit_20261027/verified.json", result)))


if __name__ == "__main__":
    main()
