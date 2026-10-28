"""Independent aggregate recomputation for all fixed saved-state model predictions."""

import json

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require


def main():
    root = REPO / "outputs/joint_limit_response_all_20261027"
    result_ref, registration_ref = reference(root / "verified.json"), reference(root / "registration.json")
    result, registration = read(result_ref), read(registration_ref)
    require(result["complete"] is result["passed"] is True, "Complete producer computation required")
    require(registration["cases"] == list(range(7)) and len(result["cases"]) == 7, "All seven fixed cases required")
    for ref in registration["source_refs"]:
        checked(ref)
    prepared = read(registration["prepared_ref"])
    comparisons = []
    for index, case in enumerate(result["cases"]):
        require(case["case"] == prepared["cases"][index]["case"], "Case identity changed")
        rows = read(case["rows_ref"])
        records = case["records"]
        require(len(records) == len(rows) == 40 and [r["index"] for r in records] == list(range(40)), "Aligned40 rows")
        metrics = {}
        for method in ("free", "limited"):
            error = np.asarray([r[f"{method}_error"] for r in records])
            require(error.shape == (40, 6) and np.isfinite(error).all(), "Finite full pose errors")
            position = float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1))))
            rotation = float(np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1))))
            np.testing.assert_allclose([position, rotation], list(case["metrics"][method].values()), atol=1e-15, rtol=0)
            metrics[method] = dict(position_rmse_m=position, rotation_rmse_rad=rotation)
        comparisons.append(
            dict(case=case["case"], metrics=metrics, projected_periods=sum(r["projected_steps"] > 0 for r in records))
        )
    require(result["physics_steps"] == result["model_calls"] == result["fresh_cases"] == 0, "Scope differs")
    print(
        json.dumps(
            publish(
                REPO / "outputs/joint_limit_response_all_verified_20261027/verified.json",
                dict(
                    complete=True,
                    passed=True,
                    result_ref=result_ref,
                    registration_ref=registration_ref,
                    cases=comparisons,
                    source_ref=reference(__file__),
                    periods=280,
                    supplemental_qp_source_ref=reference(REPO / "src/s2a/nullspace_convex_qp.py"),
                    full_launch_dependency_roster_registered=False,
                    scope="registered producer source and aggregate arithmetic audit; not independent native prediction replay",
                    control_qualified=False,
                    fresh_authorized=False,
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
