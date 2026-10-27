"""117 read-only complete local-response error decomposition before/after coverage."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/117_condition_span_error_attribution_20261025.md")
    validations = []
    for name in ("joint_wrench_condition_forward", "condition_span_forward"):
        root = Path(f"outputs/{name}_20261025")
        closure = e.json(root / "closure_verified.json")
        assert closure["complete"] and closure["numerically_verified"] and closure["owner_normal_wait_verified"]
        ref = closure["validation_ref"]
        e.check(ref["path"], ref["sha256"])
        validations.append(e.json(ref["path"]))
    before, after = validations
    assert [r["anchor_group"] for r in before["local_response"]] == list(range(6))
    assert [r["anchor_group"] for r in after["local_response"]] == list(range(6))
    results = []
    for old, new in zip(before["local_response"], after["local_response"], strict=True):
        channels = {}
        for channel in ("position", "linear_velocity", "so3_increment", "angular_velocity"):
            a, b = old["metrics"][channel], new["metrics"][channel]
            for key in ("measured", "nominal"):
                np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-12)
            item = {}
            for label, saved in (("109", a), ("116", b)):
                error = np.asarray(saved["predicted"]) - np.asarray(saved["measured"])
                columns = np.sum(error**2, axis=0)
                rows = np.sum(error**2, axis=1)
                np.testing.assert_allclose(np.sqrt(columns.sum()), saved["fitted_frobenius_error"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(rows.sum(), columns.sum(), rtol=0, atol=1e-12)
                nominal = np.asarray(saved["nominal"]) - np.asarray(saved["measured"])
                np.testing.assert_allclose(
                    np.linalg.norm(nominal), saved["nominal_frobenius_error"], rtol=0, atol=1e-12
                )
                item[label] = dict(
                    command_axis_sse=columns.tolist(),
                    response_axis_sse=rows.tolist(),
                    column_error=np.sqrt(columns).tolist(),
                    frobenius_error=float(np.linalg.norm(error)),
                )
            item["command_axis_sse_change"] = (
                np.array(item["116"]["command_axis_sse"]) - np.array(item["109"]["command_axis_sse"])
            ).tolist()
            item["nominal_error"] = b["nominal_frobenius_error"]
            channels[channel] = item
        results.append(dict(anchor_group=old["anchor_group"], channels=channels))
    e.recheck()
    output = Path("outputs/condition_span_error_attribution_20261025")
    output.mkdir(exist_ok=False)
    write_once(
        output / "result.json",
        dict(
            complete=True,
            diagnostic_only=True,
            groups=results,
            new_fits=0,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
            source_sha256=e.hashes,
        ),
    )
    print(results[4])


if __name__ == "__main__":
    main()
