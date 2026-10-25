"""Admit the fixed058 numerical result for current-only sampling preparation.

JSON and SHA checks only; no training or validation future arrays are decoded.
Numerical admission does not grant sampling or task-performance qualification.
"""

from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import read, require
from s2a.terminal_readout_policy import TerminalReadoutPolicy
from s2a.wam_clean_readout import CleanResidualReadout

ROOT = REPO / "outputs/terminal_solver_convergence_20261025"


def candidate_head(verification_ref, *, read_json=read):
    require(Path(verification_ref["path"]) == ROOT / "fit_verified.json", "Registered058 proof required")
    verified = read_json(verification_ref)
    require(
        verified["passed"] is verified["complete"] is True
        and verified["requests"] == 300
        and verified["rows"] == 12000
        and verified["performance_qualified"] is False
        and verified["position_relative_gradient_residual"] <= 1e-7
        and -1e-10 <= verified["opening_gap"] <= 2e-4,
        "Complete058 independent numerical qualification required",
    )
    require(Path(verified["producer_ref"]["path"]) == ROOT / "fit_result.json", "Registered058 producer required")
    produced = read_json(verified["producer_ref"])
    require(
        produced["complete"] is produced["passed"] is True
        and produced["producer_fits"] == 1
        and produced["cumulative_producer_attempts"] == 2
        and produced["params_ref"] == verified["params_ref"]
        and Path(produced["params_ref"]["path"]) == ROOT / "parameters.json"
        and produced["independent_numerical_qualification"] is False,
        "Fixed058 producer identity differs",
    )
    position, history = produced["position"], produced["opening_history"]
    require(
        position["cg_info"] == 0
        and 0 <= position["iterations"] <= 1000
        and position["relative_normal_residual"] <= 1e-7,
        "058 position solve budget or residual differs",
    )
    require(
        len(history) == 100
        and all(
            h["outer_step"] == i + 1 and h["cg_info"] == 0 and 0 <= h["iterations"] <= 1000
            for i, h in enumerate(history)
        ),
        "058 fixed100 IRLS budget required",
    )
    require(
        history[-1]["feasible"] is True
        and -1e-10 <= history[-1]["gap"] <= 2e-4
        and abs(history[-1]["gap"] - verified["opening_gap"]) < 1e-10,
        "058 independent original-L1 certificate differs",
    )
    for result in (produced, verified):
        require(
            all(result[k] == 0 for k in ("new_model_calls", "physics_steps", "fresh_cases")), "058 unregistered work"
        )
    prepared = read_json(produced["prepared_ref"])
    baseline_ref = prepared["params_ref"]
    require(
        baseline_ref["sha256"] == "ca01b9d8a49f2f1585c917f492e52a0a89cdb0b447f6bd2beee050ed0d2fccc0",
        "Original052 identity required",
    )
    baseline = CleanResidualReadout.from_parameters(read_json(baseline_ref))
    terminal = CleanResidualReadout.from_parameters(read_json(produced["params_ref"]))
    require(np.isfinite(terminal.coefficients).all(), "Nonfinite058 coefficients")
    return TerminalReadoutPolicy(baseline, terminal), produced["params_ref"]
