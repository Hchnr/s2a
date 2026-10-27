"""204 independent full forward-map and guard audit on all164 saved current states."""

import json

import numpy as np
from audit_wam_position import Evidence
from guarded_loading_support_independent import verify_support_inverse

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish
from s2a.tracking_guarded_loading_inverse import tracking_guarded_loading_translation
from s2a.training_support import TrainingSupport


def main():
    out = REPO / "outputs/guarded_loading_saved_states_20261025"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    for p in (
        __file__,
        "src/s2a/tracking_guarded_loading_inverse.py",
        "src/s2a/tracking_guarded_rate.py",
        "scripts/guarded_loading_support_independent.py",
        "scripts/tracking_guard_rate_independent.py",
    ):
        e.read(p)
    config = e.json("configs/qualified_inverse_rate_loading_20261025.json")
    ref = config["arm_forward_parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    params = e.json(ref["path"])
    ref = config["arm_support_calibration_ref"]
    e.check(ref["path"], ref["sha256"])
    calibration = e.json(ref["path"])
    ref = calibration["arrays_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as a:
        points = a["points"].copy()
    support = TrainingSupport(points, calibration["radius"])
    case = REPO / "outputs/qualified_inverse_rate_loading_20261025/case00"
    assert e.json(case / "independent_verified.json")["mechanical_labels_images_passed"]
    path = next(case.rglob("steps.jsonl"))
    rows = [json.loads(line) for line in e.read(path).decode().splitlines()]
    contract = e.json(path.with_name("controller.json"))["contract"]
    bounds = {k: np.asarray(contract[k])[:3] for k in ("input_min", "input_max", "output_min", "output_max")}
    decisions = []
    for row in rows:
        data = {k: np.asarray(v) for k, v in row["qualified_forward_input"].items()}
        target = np.asarray(row["desired_site"])[:3, 3] - data["x_site_pose"][0, :3, 3]
        velocity = np.asarray(row["site_velocity_reference"]["velocity"])
        previous = data["x_previous_command"][0, :3]
        physical = np.asarray(row["qualified_forward_inverse"]["physical_candidate"])
        route = row["gripper"]["route"]
        result = tracking_guarded_loading_translation(
            data,
            params,
            target,
            velocity,
            support=support,
            route=route,
            physical_command=physical,
            previous_command=previous,
            **bounds,
        )
        verify_support_inverse(
            data,
            params,
            contract,
            previous,
            target,
            velocity,
            physical,
            points,
            calibration["radius"],
            result,
            expected_route=route,
        )
        guard = result["tracking_guard"]
        decisions.append(
            dict(
                step=row["step"],
                reason=guard["reason"],
                change=guard["change"],
                feasible=guard["tracking_guard_feasible"],
                predicted_error=guard["predicted_error"],
            )
        )
    assert len(decisions) == 164
    e.recheck()
    publish(
        out / "verified.json",
        dict(
            complete=True,
            independent_forward_and_guard_verified=True,
            rows=164,
            decisions=decisions,
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
            control_qualified=False,
            limitation="One-step counterfactuals from recorded measured states, not new executed history or a closed-loop result",
        ),
    )
    print("204 all164 independently reconstructed forward/guard decisions passed;physics0")


if __name__ == "__main__":
    main()
