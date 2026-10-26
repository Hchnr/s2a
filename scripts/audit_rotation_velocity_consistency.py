"""118 fixed time-relation diagnostics; no fitted or selected projection."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/118_rotation_velocity_consistency_20261025.md")
    results = []
    for model in ("joint_wrench_condition_forward", "condition_span_forward"):
        root = Path(f"outputs/{model}_20261025")
        c = e.json(root / "closure_verified.json")
        assert c["complete"] and c["numerically_verified"] and c["owner_normal_wait_verified"]
        ref = c["validation_ref"]
        e.check(ref["path"], ref["sha256"])
        v = e.json(ref["path"])
        assert [r["anchor_group"] for r in v["local_response"]] == list(range(6))
        for row in v["local_response"]:
            m = row["metrics"]
            for name in ("measured", "predicted", "nominal"):
                rotation = np.asarray(m["so3_increment"][name])
                velocity = np.asarray(m["angular_velocity"][name])
                assert rotation.shape == velocity.shape == (3, 3)
                assert np.all(np.isfinite(rotation)) and np.all(np.isfinite(velocity))
                results.append(
                    dict(
                        model=model,
                        group=row["anchor_group"],
                        kind=name,
                        rotation_norm=float(np.linalg.norm(rotation)),
                        dt_endpoint_velocity_norm=float(np.linalg.norm(0.05 * velocity)),
                        endpoint_constant_approximation_error=float(np.linalg.norm(rotation - 0.05 * velocity)),
                        linear_velocity_change_approximation_error=float(np.linalg.norm(rotation - 0.025 * velocity)),
                    )
                )
    e.recheck()
    output = Path("outputs/rotation_velocity_consistency_20261025")
    output.mkdir(exist_ok=False)
    write_once(
        output / "result.json",
        dict(
            complete=True,
            diagnostic_only=True,
            exact_integration_verified=False,
            results=results,
            source_sha256=e.hashes,
            new_fits=0,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )
    for r in results:
        if r["group"] == 4:
            print(r)


if __name__ == "__main__":
    main()
