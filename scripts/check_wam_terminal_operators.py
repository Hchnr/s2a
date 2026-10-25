"""Full training-cache operator qualification, independent per-trajectory loss."""

import os
from pathlib import Path

import numpy as np
from verify_wam_clean_training_capture import Evidence, require

from s2a.clean_training_runtime import publish, reference
from s2a.terminal_trajectory_operators import operators
from s2a.trajectory_objective import opening_loss, position_loss


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    root = Path("outputs/terminal_trajectory_objective_20261025")
    require(not (root / "operator_qualification.json").exists(), "New output required")
    e = Evidence()
    prepared = e.json(reference(root / "prepared.json"))
    require(
        prepared["passed"] and prepared["requests"] == 300 and prepared["rows"] == 12000,
        "Complete preparation required",
    )
    for ref in prepared["source_refs"]:
        e.path(ref)
    e.path(prepared["plan_ref"])
    with np.load(e.path(prepared["cache_ref"]), allow_pickle=False) as z:
        c = {k: z[k].copy() for k in z.files}
    require(all(np.isfinite(v).all() for v in c.values()), "Finite full cache required")
    require(c["phi"].shape == (300, 40, 1291), "Exact plan057 dimensions required")
    ap, bp, ag, bg, w = operators(c)
    rng = np.random.default_rng(42)
    delta = rng.normal(size=c["coefficients052"].shape) * 1e-4
    physical = (c["raw_clean"] + c["phi"] @ (c["coefficients052"] + delta) - c["action_bias"]) / c["action_scale"]
    pl, gl = 0.0, 0.0
    gp, gg = np.zeros((1291, 3)), np.zeros(1291)
    for i in range(300):
        loss, grad = position_loss(
            physical[i, :, :3],
            c["before_rotation"][i],
            c["initial_position"][i],
            c["desired_positions"][i],
            c["endpoint_weights"],
        )
        pl += loss / 300 / 0.05**2
        gp += c["phi"][i].T @ (grad / c["action_scale"][:3]) / 300 / 0.05**2
        loss, grad = opening_loss(physical[i, :, 9], c["desired_opening"][i], c["endpoint_weights"])
        gl += loss / 300
        gg += c["phi"][i].T @ (grad / c["action_scale"][9]) / 300
    ep, eg = ap.matvec(delta[:, :3].ravel()) + bp, ag.matvec(delta[:, 9]) + bg
    checks = dict(
        position_loss_error=abs(ep @ ep - pl),
        opening_loss_error=abs(w @ abs(eg) - gl),
        position_gradient_max_error=float(np.max(abs(2 * ap.rmatvec(ep) - gp.ravel()))),
        opening_gradient_max_error=float(np.max(abs(ag.rmatvec(w * np.sign(eg)) - gg))),
    )
    for name, a in (("position", ap), ("opening", ag)):
        x, y = rng.normal(size=a.shape[1]), rng.normal(size=a.shape[0])
        lhs, rhs = a.matvec(x) @ y, x @ a.rmatvec(y)
        require(np.isclose(lhs, rhs, rtol=1e-10, atol=1e-10), "Adjoint identity failed")
        checks[name + "_adjoint_relative_error"] = float(abs(lhs - rhs) / max(1, abs(lhs), abs(rhs)))
    require(all(value < 1e-9 for value in checks.values()), "Independent complete objective/gradient mismatch")
    e.finish()
    publish(
        root / "operator_qualification.json",
        dict(
            passed=True,
            complete=True,
            seed=42,
            requests=300,
            rows=12000,
            prepared_ref=reference(root / "prepared.json"),
            checks=checks,
            input_sha256=e.inputs,
            source_refs=[
                reference(Path(__file__)),
                reference(Path("src/s2a/terminal_trajectory_operators.py")),
                reference(Path("src/s2a/trajectory_objective.py")),
            ],
            producer_fits=0,
            new_model_calls=0,
            physics_steps=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
