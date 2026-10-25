"""One separately registered plan058 convergence fit; numerical qualification remains a separate result."""

import os
from pathlib import Path

import numpy as np
from verify_wam_clean_training_capture import Evidence, require

from s2a.clean_training_runtime import publish, reference
from s2a.terminal_objective_solver import fit_opening_lad, ridge_quadratic
from s2a.terminal_trajectory_operators import operators


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    root = Path("outputs/terminal_solver_convergence_20261025")
    previous_root = Path("outputs/terminal_trajectory_objective_20261025")
    require(not root.exists(), "New058 namespace required")
    root.mkdir()
    e = Evidence()
    prior = e.json(reference(previous_root / "fit_result.json"))
    prior_verified = e.json(reference(previous_root / "fit_verified.json"))
    e.json(reference(previous_root / "fit_failure.json"))
    require(
        prior["passed"] is prior_verified["passed"] is False and prior["complete"] and prior_verified["complete"],
        "Original057 complete failure required",
    )
    q = e.json(reference(previous_root / "operator_qualification.json"))
    require(q["passed"] and q["complete"] and q["requests"] == 300, "Full operator qualification required")
    for ref in q["source_refs"]:
        e.path(ref)
    p = e.json(q["prepared_ref"])
    for ref in p["source_refs"] + [p["plan_ref"]]:
        e.path(ref)
    params = e.json(p["params_ref"])
    with np.load(e.path(p["cache_ref"]), allow_pickle=False) as z:
        c = {k: z[k].copy() for k in z.files}
    require(np.array_equal(c["coefficients052"], np.asarray(params["coefficients"])), "Baseline identity differs")
    sources = [
        reference(Path(__file__)),
        reference(Path("src/s2a/terminal_objective_solver.py")),
        reference(Path("src/s2a/terminal_trajectory_operators.py")),
        reference(Path("scripts/verify_wam_terminal_convergence.py")),
        reference(Path("plans/v0.0.1/058_terminal_solver_convergence_20261025.md")),
    ]
    for ref in sources:
        e.path(ref)
    publish(
        root / "fit_attempt.json",
        dict(
            attempt=1,
            cumulative_producer_attempt=2,
            pid=os.getpid(),
            pid_birth=Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19],
            source_refs=sources,
            prepared_ref=q["prepared_ref"],
        ),
    )
    try:
        ap, bp, ag, bg, w = operators(c)
        ridge = 0.1 / 1290
        dp, position = ridge_quadratic(ap, bp, ridge, maxiter=1000)
        print(f"position: {position}", flush=True)
        dg, opening = fit_opening_lad(ag, bg, w, ridge, epsilon=1e-4, outer_steps=100, maxiter=1000)
        print(f"opening final: {opening[-1]}", flush=True)
        coef = c["coefficients052"].copy()
        coef[:, :3] += dp.reshape(1291, 3)
        coef[:, 9] += dg
        params["coefficients"] = coef.tolist()
        e.finish()
        publish(root / "parameters.json", params)
        passed = (
            position["cg_info"] == 0
            and position["relative_normal_residual"] <= 1e-7
            and len(opening) == 100
            and all(v["cg_info"] == 0 for v in opening)
            and opening[-1]["feasible"]
            and -1e-10 <= opening[-1]["gap"] <= 2e-4
        )
        publish(
            root / "fit_result.json",
            dict(
                passed=bool(passed),
                complete=True,
                producer_fits=1,
                cumulative_producer_attempts=2,
                independent_numerical_qualification=False,
                prepared_ref=q["prepared_ref"],
                qualification_ref=reference(previous_root / "operator_qualification.json"),
                params_ref=reference(root / "parameters.json"),
                position=position,
                opening_history=opening,
                input_sha256=e.inputs,
                source_refs=sources,
                new_model_calls=0,
                physics_steps=0,
                fresh_cases=0,
            ),
        )
        require(passed, "Fixed-budget fit failed; preserve result, no retry")
    except BaseException as error:
        publish(root / "fit_failure.json", dict(passed=False, error=repr(error), producer_fit_attempts=1))
        raise


if __name__ == "__main__":
    main()
