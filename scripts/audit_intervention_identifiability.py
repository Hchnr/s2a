"""114 distinguish regularized unique solutions from intervention identifiability."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/114_intervention_identifiability_20261025.md")
    root = Path("outputs/joint_wrench_condition_forward_20261025")
    closed = e.json(root / "closure_verified.json")
    assert closed["complete"] and closed["numerically_verified"] and closed["owner_normal_wait_verified"]
    fit = e.json(root / "fit_result.json")
    for key in ("parameter_ref", "cache_ref"):
        ref = fit[key]
        e.check(ref["path"], ref["sha256"])
    p = e.json(fit["parameter_ref"]["path"])
    with np.load(fit["cache_ref"]["path"], allow_pickle=False) as cache:
        x = cache["features"][4865:].copy()
    assert x.shape == (135, 1128) and np.max(np.abs(x[:, :282])) == 0

    def svd(matrix):
        _, values, right = np.linalg.svd(matrix, full_matrices=False)
        tolerance = max(matrix.shape) * np.finfo(float).eps * values[0]
        rank = int(np.sum(values > tolerance))
        return rank, values, right[:rank], tolerance

    slope_rank, singular, _, threshold = svd(x[:, 282:])
    phi = []
    axis_ranks = []
    for axis in range(3):
        rows = [i for i, r in enumerate(fit["contrast_sources"]) if r["axis"] == axis]
        block = x[rows, (axis + 1) * 282 : (axis + 2) * 282]
        assert block.shape == (45, 282) and np.all(block[:, 0] != 0)
        condition = block / block[:, :1]
        phi.append(condition)
        axis_ranks.append(svd(condition)[0])
    np.testing.assert_allclose(phi[0], phi[1], rtol=0, atol=1e-10)
    np.testing.assert_allclose(phi[0], phi[2], rtol=0, atol=1e-10)
    rank, condition_singular, basis, tolerance = svd(phi[0])
    audit = e.json("outputs/intervention_condition_coverage_20261025/result.json")
    development = []
    for row in audit["development_conditions"]:
        z = (np.asarray(row["raw"]) - p["state_mean"]) / p["state_scale"]
        phase = np.asarray(p["frequencies"]) @ z[:83]
        joint = np.asarray(p["joint_frequencies"]) @ z
        query = np.r_[
            1,
            z[:83],
            np.cos(phase) / 8,
            np.sin(phase) / 8,
            z[83:],
            np.cos(joint) / np.sqrt(32),
            np.sin(joint) / np.sqrt(32),
        ]
        assert query.shape == (282,)
        residual = query - (query @ basis.T) @ basis
        development.append(
            dict(
                anchor_group=row["anchor_group"],
                used_for_original_contrast_fit=row["intervention_used_for_fit"],
                unspanned_feature_norm_fraction=float(np.linalg.norm(residual) / np.linalg.norm(query)),
            )
        )
    e.recheck()
    out = Path("outputs/intervention_identifiability_20261025")
    out.mkdir(exist_ok=False)
    write_once(
        out / "result.json",
        dict(
            complete=True,
            slope_columns=846,
            intervention_equations=135,
            slope_rank=slope_rank,
            slope_rank_tolerance=float(threshold),
            slope_singular_values=singular.tolist(),
            condition_columns=282,
            condition_rank=rank,
            per_axis_condition_ranks=axis_ranks,
            condition_rank_tolerance=float(tolerance),
            condition_singular_values=condition_singular.tolist(),
            development_condition_projection=development,
            source_sha256=e.hashes,
            parameter_fits=0,
            physics_steps=0,
            diagnostic_only=True,
        ),
    )
    print("114 intervention slope rank", slope_rank, "/846;condition rank", rank, "/282")
    print(development)


if __name__ == "__main__":
    main()
