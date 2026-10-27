"""Read-only paired physical tracking diagnosis for the first SQP case."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from scipy.spatial.transform import Rotation

from s2a.durable_stage_supervisor import publish


def pose_error(desired, actual):
    return np.r_[
        desired[:3, 3] - actual[:3, 3],
        Rotation.from_matrix(desired[:3, :3] @ actual[:3, :3].T).as_rotvec(),
    ]


def metrics(errors, *, start_step=0):
    norms = np.column_stack((np.linalg.norm(errors[:, :3], axis=1), np.linalg.norm(errors[:, 3:], axis=1)))
    return dict(
        position_rmse_m=float(np.sqrt(np.mean(norms[:, 0] ** 2))),
        so3_rmse_rad=float(np.sqrt(np.mean(norms[:, 1] ** 2))),
        maximum_position_m=float(norms[:, 0].max()),
        maximum_so3_rad=float(norms[:, 1].max()),
        maximum_position_step=start_step + int(norms[:, 0].argmax()),
        maximum_so3_step=start_step + int(norms[:, 1].argmax()),
    )


def main():
    import json

    output = Path("outputs/full_coupled_sqp_case00_tracking_20261026/verified.json")
    assert not output.exists()
    e = Evidence()
    e.read(__file__)
    prefix = e.json("outputs/full_coupled_sqp_prefix_03_20261026.json")
    assert prefix["complete"] and prefix["verified_cases"] == 3
    results, all_rows = {}, {}
    for label, name in (
        ("prior232", "current_coupled_control_20261026"),
        ("sqp244", "full_coupled_sqp_control_20261026"),
    ):
        root = Path("outputs") / name / "case00"
        verified = e.json(root / "independent_verified.json")
        assert verified["complete"] and verified["mechanical_labels_images_passed"]
        case = next(root.glob("simulation/*/*"))
        rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
        assert len(rows) == 164 and [r["step"] for r in rows] == list(range(164))
        controller = e.json(case / "controller.json")
        config = e.json(root / "execution_config.json")
        template = e.json(config["policy_template"])
        tool = np.asarray(e.json(template["tool_contract"])["source_from_tcp"])
        errors = np.array(
            [pose_error(np.asarray(r["desired_tcp"]), np.asarray(r["after_source"]) @ tool) for r in rows]
        )
        np.testing.assert_allclose(errors, [r["tcp_error"] for r in rows], atol=1e-9, rtol=0)
        summary = metrics(errors)
        for key, expected in verified["control"]["tracking"]["admitted"].items():
            np.testing.assert_allclose(summary[key], expected, atol=1e-10, rtol=0)
        command = np.asarray([r["command"][:6] for r in rows])
        changes = np.diff(np.vstack((np.zeros((1, 6)), command)), axis=0)
        assert np.max(np.abs(changes)) <= 0.25 + 1e-9
        norms = np.linalg.norm(errors[:, :3], axis=1)
        angles = np.linalg.norm(errors[:, 3:], axis=1)
        windows = []
        for start, end in ((0, 40), (40, 80), (80, 120), (120, 164)):
            windows.append(dict(start=start, end_exclusive=end, **metrics(errors[start:end], start_step=start)))
        results[label] = dict(
            tracking=summary,
            windows=windows,
            first_over_5mm=next((int(i) for i in np.flatnonzero(norms > 0.005)), None),
            first_over_50mrad=next((int(i) for i in np.flatnonzero(angles > 0.05)), None),
            rate_limited_periods=sum(r["rate_limited"] for r in rows),
            saturated_periods=sum(r["saturated"] for r in rows),
            command_increment_rms=float(np.sqrt(np.mean(changes**2))),
            outcome=verified["control"]["outcome"],
        )
        if label == "sqp244":
            source_from_site = np.asarray(controller["source_from_site"])
            proposal_errors = np.array(
                [
                    pose_error(
                        np.asarray(r["full_pose_inverse"]["predicted_pose"]),
                        np.asarray(r["after_source"]) @ source_from_site,
                    )
                    for r in rows
                ]
            )
            results[label]["continuous_proposal_prediction_error"] = metrics(proposal_errors)
            selected = sorted(
                {summary["maximum_position_step"], summary["maximum_so3_step"]}
                | set(
                    range(max(0, summary["maximum_position_step"] - 3), min(164, summary["maximum_position_step"] + 4))
                )
            )
            results[label]["selected_periods"] = [
                dict(
                    step=i,
                    tcp_error=errors[i].tolist(),
                    continuous_prediction_error=proposal_errors[i].tolist(),
                    command=rows[i]["command"],
                    goal_mode=rows[i]["full_pose_inverse"]["goal_mode"],
                    objective=rows[i]["full_pose_inverse"]["objective"],
                    innovation_before=rows[i]["coupled_response_history"]["innovation_before"],
                    opening_before=rows[i]["opening_before"],
                    success=rows[i]["success"],
                )
                for i in selected
            ]
        all_rows[label] = rows
    for key in ("desired_tcp", "desired_site", "opening_target"):
        np.testing.assert_allclose(
            [r[key] for r in all_rows["prior232"]], [r[key] for r in all_rows["sqp244"]], atol=1e-9, rtol=0
        )
    e.recheck()
    publish(
        output,
        dict(
            complete=True,
            scope="paired completed physical case00;descriptive tracking/prediction diagnostics;no unique causal claim",
            results=results,
            reference_values_equal=True,
            source_sha256=e.hashes,
            new_physics_steps=0,
            model_calls=0,
            fits=0,
            fresh_cases=0,
        ),
    )
    for label, result in results.items():
        print(label, result["tracking"], "first5mm", result["first_over_5mm"], "rate", result["rate_limited_periods"])


if __name__ == "__main__":
    main()
