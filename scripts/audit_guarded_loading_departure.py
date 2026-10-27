"""206 independently checked case03: separate predictor error and guard feasibility."""

import json

import numpy as np
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    out = REPO / "outputs/guarded_loading_departure_20261025"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    e.read(__file__)
    case = REPO / "outputs/qualified_inverse_guarded_loading_20261025/case03"
    proof = e.json(case / "independent_verified.json")
    assert proof["mechanical_labels_images_passed"]
    path = next(case.rglob("steps.jsonl"))
    rows = [json.loads(line) for line in e.read(path).decode().splitlines()]
    mapping = np.asarray(e.json(path.with_name("controller.json"))["source_from_site"])
    records = []
    for r in rows:
        solved = r["qualified_forward_inverse"]
        guard = solved["tracking_guard"]
        current = np.asarray(r["qualified_forward_input"]["x_site_pose"])[0]
        after = np.asarray(r["after_source"]) @ mapping
        actual = after[:3, 3] - current[:3, 3]
        model = np.asarray(solved["predicted_motion"])
        velocity_error = 0.05 * (
            np.asarray(solved["predicted_linear_velocity"]) - r["site_velocity_reference"]["velocity"]
        )
        target = np.asarray(r["desired_site"])[:3, 3] - current[:3, 3]
        assert abs(r["actual_dt"] - 0.05) < 1e-10
        records.append(
            dict(
                step=r["step"],
                reason=guard["reason"],
                feasible=guard["tracking_guard_feasible"],
                change=guard["change"],
                predicted_position_error_m=float(np.linalg.norm(model - target)),
                predicted_scaled_velocity_error_m=float(np.linalg.norm(velocity_error)),
                position_prediction_error_m=float(np.linalg.norm(actual - model)),
                actual_site_error_m=float(np.linalg.norm(actual - target)),
            )
        )
    window = [r for r in records if 38 <= r["step"] <= 65]
    e.recheck()
    publish(
        out / "verified.json",
        dict(
            complete=True,
            rows=records,
            max_window_position_prediction_error_m=max(r["position_prediction_error_m"] for r in window),
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
            control_qualified=False,
            limitation="Observed temporal association only; neither proof of unique failure cause nor hypothetical new-controller rollout",
        ),
    )
    for i in [38, 45, 50, 55, 60, 65, 66, 67, 68, 70]:
        print(records[i])


if __name__ == "__main__":
    main()
