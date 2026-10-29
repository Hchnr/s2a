"""Check a null-space solution of the exact captured QP under original KKT gates."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.nullspace_convex_qp import solve_qp as nullspace_solve
from s2a.strict_convex_qp import solve_qp as original_solve


def main():
    capture_ref = reference(REPO / "outputs/tracking_wam_qp_failure_capture_v2_20261027/result.json")
    capture = read(capture_ref)
    require(
        capture["complete"] is capture["passed"] is capture["all_prefix_endpoint_sha_equal"] is True,
        "Exact physical prefix capture",
    )
    with np.load(checked(capture["qp_ref"]), allow_pickle=False) as archive:
        values = [archive[k].copy() for k in ["hessian", "linear", "constraints", "bounds", "initial"]]
    h, g, a, b, initial = values
    try:
        original_solve(*values)
    except RuntimeError as error:
        require("Dependent QP blocking constraint" in str(error), "Same original failure")
        original_error = str(error)
    else:
        raise ValueError("Original failure not reproduced")
    solved = nullspace_solve(*values)
    x = np.asarray(solved["solution"])
    dual = np.zeros(len(a))
    dual[np.asarray(solved["active"], dtype=int)] = solved["multipliers"]
    slack = b - a @ x
    stationarity = float(np.max(abs(h @ x + g + a.T @ dual)))
    complementarity = float(np.max(abs(dual * slack)))
    primal = max(0.0, float(-slack.min()))
    require(np.allclose(h, h.T, atol=1e-12, rtol=1e-12), "Symmetric objective")
    np.linalg.cholesky(h)
    require(
        primal <= 1e-9 and stationarity <= 1e-8 and complementarity <= 1e-8 and dual.min() >= -1e-9,
        "Original independent primal/dual/KKT gates",
    )
    delta = x - initial
    objective_delta = float(delta @ (h @ initial + g) + 0.5 * delta @ h @ delta)
    require(objective_delta <= 1e-9, "No increase over feasible initial sequence")
    root = REPO / "outputs/tracking_wam_captured_qp_check_20261027"
    root.mkdir(exist_ok=False)
    with (root / "solution.npz").open("xb") as stream:
        np.savez_compressed(stream, solution=x, dual=dual)
    print(
        json.dumps(
            publish(
                root / "verified.json",
                dict(
                    complete=True,
                    passed=True,
                    capture_ref=capture_ref,
                    qp_ref=capture["qp_ref"],
                    solution_ref=reference(root / "solution.npz"),
                    source_refs=[
                        reference(Path(__file__)),
                        reference(REPO / "src/s2a/nullspace_convex_qp.py"),
                        reference(REPO / "src/s2a/strict_convex_qp.py"),
                    ],
                    original_error=original_error,
                    primal_violation=primal,
                    stationarity=stationarity,
                    complementarity=complementarity,
                    minimum_dual=float(dual.min()),
                    objective_delta_from_feasible_initial=objective_delta,
                    iterations=solved["iterations"],
                    physics_steps=0,
                    world_model_calls=0,
                    fresh_cases=0,
                    scope="Same captured strictly convex QP and original numerical gates only; no new dispatched action or task qualification",
                ),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
