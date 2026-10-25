"""Independent complete per-trajectory gradient and original-L1 dual proof."""

import os
from pathlib import Path

import numpy as np
from verify_wam_clean_training_capture import Evidence, require

from s2a.clean_training_runtime import publish, reference
from s2a.trajectory_objective import position_loss


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    root = Path("outputs/terminal_trajectory_objective_20261025")
    e = Evidence()
    result = e.json(reference(root / "fit_result.json"))
    require(result["complete"] and result["producer_fits"] == 1, "Complete producer result required")
    for ref in result["source_refs"]:
        e.path(ref)
    p = e.json(result["prepared_ref"])
    for ref in p["source_refs"] + [p["plan_ref"]]:
        e.path(ref)
    params = e.json(result["params_ref"])
    baseline = e.json(p["params_ref"])
    with np.load(e.path(p["cache_ref"]), allow_pickle=False) as z:
        c = {k: z[k].copy() for k in z.files}
    coef, old = np.asarray(params["coefficients"]), np.asarray(baseline["coefficients"])
    require(
        np.array_equal(coef[:, 3:9], old[:, 3:9])
        and all(params[k] == baseline[k] for k in ("center", "scale", "ridge", "schema_version")),
        "Frozen readout columns/layout changed",
    )
    dp, dg = coef[:, :3] - old[:, :3], coef[:, 9] - old[:, 9]
    ridge = 0.1 / 1290
    gp, gp0 = np.zeros((1291, 3)), np.zeros((1291, 3))
    dual_adjoint = np.zeros(1291)
    primal = dual_base = position = 0.0
    for i in range(300):
        phi = c["phi"][i]
        physical = (c["raw_clean"][i] + phi @ coef - c["action_bias"]) / c["action_scale"]
        base = (c["raw_clean"][i] + phi @ old - c["action_bias"]) / c["action_scale"]
        args = (c["before_rotation"][i], c["initial_position"][i], c["desired_positions"][i], c["endpoint_weights"])
        loss, grad = position_loss(physical[:, :3], *args)
        position += loss / 300 / 0.05**2
        gp += phi.T @ (grad / c["action_scale"][:3]) / 300 / 0.05**2
        _, grad0 = position_loss(base[:, :3], *args)
        gp0 += phi.T @ (grad0 / c["action_scale"][:3]) / 300 / 0.05**2
        error = physical[:, 9] - c["desired_opening"][i]
        base_error = base[:, 9] - c["desired_opening"][i]
        weights = c["endpoint_weights"] / 300
        dual = weights * error / np.sqrt(error**2 + 1e-8)
        require((abs(dual) <= weights).all(), "Infeasible L1 dual")
        dual_adjoint += phi.T @ dual / c["action_scale"][9]
        primal += weights @ abs(error)
        dual_base += dual @ base_error
    residual = float(np.linalg.norm(gp + 2 * ridge * dp) / max(np.linalg.norm(gp0), np.finfo(float).tiny))
    primal += ridge * (dg @ dg)
    lower = dual_base - (dual_adjoint @ dual_adjoint) / (4 * ridge)
    gap = float(primal - lower)
    passed = result["passed"] and residual <= 1e-7 and -1e-10 <= gap <= 2e-4
    e.finish()
    publish(
        root / "fit_verified.json",
        dict(
            passed=bool(passed),
            complete=True,
            producer_ref=reference(root / "fit_result.json"),
            params_ref=result["params_ref"],
            requests=300,
            rows=12000,
            position_relative_gradient_residual=residual,
            position_regularized_objective=float(position + ridge * np.sum(dp**2)),
            opening_primal=float(primal),
            opening_dual=float(lower),
            opening_gap=gap,
            input_sha256=e.inputs,
            source_ref=reference(Path(__file__)),
            new_fits=0,
            new_model_calls=0,
            physics_steps=0,
            fresh_cases=0,
            performance_qualified=False,
        ),
    )
    require(passed, "Independent terminal objective qualification failed")


if __name__ == "__main__":
    main()
