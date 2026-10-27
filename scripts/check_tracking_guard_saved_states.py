"""203 saved-state counterfactual guard audit; no physical rerun or fitted labels."""

import json

import numpy as np
from audit_wam_position import Evidence
from tracking_guard_rate_independent import verify_guard

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish
from s2a.tracking_guarded_rate import guarded_rate_command


def main():
    out = REPO / "outputs/tracking_guard_saved_states_20261025"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    for p in (
        __file__,
        "scripts/tracking_guard_rate_independent.py",
        "src/s2a/tracking_guarded_rate.py",
        "plans/v0.0.1/202_tracking_guarded_loading_rate_20261025.md",
    ):
        e.read(p)
    config = e.json("configs/qualified_inverse_rate_loading_20261025.json")
    ref = config["arm_support_calibration_ref"]
    e.check(ref["path"], ref["sha256"])
    radius = e.json(ref["path"])["radius"]
    case = REPO / "outputs/qualified_inverse_rate_loading_20261025/case00"
    verified = e.json(case / "independent_verified.json")
    assert verified["mechanical_labels_images_passed"]
    path = next(case.rglob("steps.jsonl"))
    rows = [json.loads(line) for line in e.read(path).decode().splitlines()]
    contract = e.json(path.with_name("controller.json"))["contract"]
    lo, hi, outlo, outhi = [np.asarray(contract[k])[:3] for k in ("input_min", "input_max", "output_min", "output_max")]
    scale = (outhi - outlo) / (hi - lo)
    results = []
    for row in rows:
        s = row["qualified_forward_inverse"]
        learned = s["learned_solution"]
        matrix = np.asarray(learned["response_jacobian"])[:3] * scale[None, :]
        offset = np.asarray(learned["response_offset"])[:3] + np.asarray(learned["response_jacobian"])[:3] @ (
            outlo - lo * scale
        )
        target = (
            np.asarray(row["desired_site"])[:3, 3] - np.asarray(row["qualified_forward_input"]["x_site_pose"])[0, :3, 3]
        )
        previous = np.asarray(row["qualified_forward_input"]["x_previous_command"])[0, :3]
        before = np.asarray(s["pre_rate_command"])
        if row["gripper"]["route"] != "native_close_load":
            results.append(dict(step=row["step"], reason="tracking_route_unchanged"))
            continue
        if s["support_decision"]["distance"] >= 2 * radius:
            results.append(dict(step=row["step"], reason="outside_training_support"))
            continue
        value = guarded_rate_command(matrix, offset, target, previous, before)
        independently = verify_guard(matrix, offset, target, previous, before, value)
        assert np.all(value["command"] >= lo - 1e-12) and np.all(value["command"] <= hi + 1e-12)
        results.append(
            dict(
                step=row["step"],
                reason="guard_evaluated",
                feasible=independently["feasible"],
                change=independently["minimal_change"],
                predicted_error_m=independently["predicted_error"],
            )
        )
    e.recheck()
    evaluated = [r for r in results if r["reason"] == "guard_evaluated"]
    publish(
        out / "verified.json",
        dict(
            complete=True,
            rows=results,
            independently_verified_rows=len(evaluated),
            feasible_rows=sum(r["feasible"] for r in evaluated),
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
            control_qualified=False,
            limitation="Only same recorded states, not closed-loop outcomes or an actual-error guarantee",
        ),
    )
    print("203 evaluated", len(evaluated), "feasible", sum(r["feasible"] for r in evaluated), "physics0")


if __name__ == "__main__":
    main()
