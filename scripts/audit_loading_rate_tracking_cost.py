"""201 same-state forward-predicted cost of final loading-rate projection."""


import numpy as np
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    out = REPO / "outputs/loading_rate_tracking_cost_20261025"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    e.read(__file__)
    case = REPO / "outputs/qualified_inverse_rate_loading_20261025/case00"
    verified = e.json(case / "independent_verified.json")
    assert verified["mechanical_labels_images_passed"]
    path = next(case.rglob("steps.jsonl"))
    import json

    rows = [json.loads(line) for line in e.read(path).decode().splitlines()]
    contract = e.json(path.with_name("controller.json"))["contract"]
    lo, hi, outlo, outhi = [np.asarray(contract[k])[:3] for k in ("input_min", "input_max", "output_min", "output_max")]
    results = []
    for row in rows:
        solved = row["qualified_forward_inverse"]
        before = np.asarray(solved["pre_rate_command"])
        target = (
            np.asarray(row["desired_site"])[:3, 3] - np.asarray(row["qualified_forward_input"]["x_site_pose"])[0, :3, 3]
        )
        physical = outlo + (before - lo) * (outhi - outlo) / (hi - lo)
        learned = solved["learned_solution"]
        predicted = np.asarray(learned["response_offset"]) + np.asarray(learned["response_jacobian"]) @ physical
        before_error = float(np.linalg.norm(predicted[:3] - target))
        after_error = float(np.linalg.norm(np.asarray(solved["predicted_motion"]) - target))
        changed = bool(np.max(np.abs(before - solved["command"])) > 1e-12)
        results.append(
            dict(
                step=row["step"],
                changed=changed,
                before_projection_predicted_error_m=before_error,
                after_projection_predicted_error_m=after_error,
                actual_source_error_m=float(np.linalg.norm(row["source_error"][:3])),
            )
        )
    changed = [r for r in results if r["changed"]]
    e.recheck()
    publish(
        out / "verified.json",
        dict(
            complete=True,
            rows=results,
            changed_periods=len(changed),
            predicted_error_increased_periods=sum(
                r["after_projection_predicted_error_m"] > r["before_projection_predicted_error_m"] for r in changed
            ),
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
            control_qualified=False,
            limitation="Same measured state affine counterfactual; not an unmodified physical rollout or proof of unique failure cause",
        ),
    )
    print(
        len(changed),
        sum(r["after_projection_predicted_error_m"] > r["before_projection_predicted_error_m"] for r in changed),
    )


if __name__ == "__main__":
    main()
