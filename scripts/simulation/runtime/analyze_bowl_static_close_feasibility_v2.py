"""Recover the fixed-pose closing search using explicit HiGHS tolerances."""

import copy
import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import analyze_bowl_static_close_feasibility as previous
from scripts.simulation.runtime import diagnose_bowl_static_close_lp as diagnostic
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/gripper/bowl_static_close_feasibility_recovery_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/gripper/bowl_static_close_feasibility_recovery_20261112.md"
OUTPUT = ROOT / "outputs/bowl_static_close_feasibility_recovery_20261112/result.json"
QA = ROOT / "outputs/qa/bowl_static_close_feasibility_recovery_20261112/result.json"
FAILURE = {
    "path": str(ROOT / "outputs/bowl_static_close_feasibility_20261112/result.json"),
    "sha256": "cba6b1b59e6b7d2c237705281ad0ca9863d09729080a843815262fb74214cdd2",
}
DIAGNOSTIC = {
    "path": str(ROOT / "outputs/bowl_static_close_lp_diagnostic_20261112/result.json"),
    "sha256": "e294dfd3f5bc63ea7adbadcd9259ea6b0d11751a8805a54ef4eacde6fe6e7655",
}
FIXED = copy.deepcopy(previous.FIXED)
FIXED.update(
    schema="s2a.bowl-static-close-feasibility-recovery.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    solver_primal_tolerance=1e-9,
    solver_dual_tolerance=1e-9,
    prior_lp_attempts=13,
    previous_failure_sha256=FAILURE["sha256"],
    tolerance_diagnostic_sha256=DIAGNOSTIC["sha256"],
)
_PINNED = (
    (
        "scripts/simulation/runtime/analyze_bowl_static_close_feasibility.py",
        "c110cc0750da8e2658c1d028e259dcc8561ef10ae1dce81aa1eab9d5b70ba938",
    ),
    (
        "src/s2a/research/native_runtime/bilateral_box_feasibility.py",
        "6744a5f868acb451c40531074de9653416eba406c1798eb090e41d23b973b8b3",
    ),
)


def solve_precise(left, right, reference_error, maximum_close):
    """Retain the original LP and acceptance rule with explicit solver precision."""
    e = np.asarray(reference_error, dtype=float)
    if e.shape != (3,) or not np.isfinite(e).all() or not np.isfinite(maximum_close) or maximum_close < 0:
        raise ValueError("Finite reference error and nonnegative joint closing range are required")
    a, b = diagnostic.linear_problem(left, right, e)
    result = linprog(
        [0, 0, 0, 0, 1],
        A_ub=a,
        b_ub=b,
        bounds=[(None, None)] * 3 + [(0, maximum_close), (0, None)],
        method="highs",
        options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9),
    )
    value = dict(status=int(result.status), iterations=int(result.nit), success=bool(result.success))
    if result.success:
        x = np.asarray(result.x, dtype=float)
        violation = max(0.0, float(np.max(a @ x - b)), float(-x[3]), float(x[3] - maximum_close), float(-x[4]))
        if not np.isfinite(x).all() or violation > 1e-8:
            raise ValueError("A finite, constraint-verified linear solution is required")
        value.update(
            translation_world_m=x[:3],
            closing_m=float(x[3]),
            infinity_error_m=float(x[4]),
            euclidean_error_m=float(np.linalg.norm(x[:3] - e)),
            maximum_constraint_violation_m=violation,
        )
    return value


def recovered_publication(path, value):
    """Preserve the original failure, diagnostic, source identities and actual LP costs."""
    prefix.require(path == OUTPUT, "The recovered result must use its own output")
    for ref in (FAILURE, DIAGNOSTIC):
        prefix.checked(ref)
    attempted = sum(branch.get("lp_attempts", 0) for branch in value["branches"].values())
    value.update(
        previous_failure_ref=FAILURE,
        tolerance_diagnostic_ref=DIAGNOSTIC,
        prior_lp_attempts=13,
        cumulative_lp_attempts=13 + attempted,
        solver_primal_tolerance=1e-9,
        solver_dual_tolerance=1e-9,
    )
    value["source_refs"].extend(
        prefix.reference(path)
        for path in (
            Path(__file__).resolve(),
            Path(diagnostic.__file__).resolve(),
            ROOT / "tests/research/native_runtime/test_analyze_bowl_static_close_feasibility_v2.py",
        )
    )
    return publish_result(path, value)


def bind():
    prefix.require(
        all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha for path, sha in _PINNED),
        "The earlier search and solver sources must remain unchanged",
    )
    failure, diag = prefix.read(FAILURE), prefix.read(DIAGNOSTIC)
    prefix.require(
        failure["complete"] is failure["passed"] is False
        and failure["branches"]["fixed"]["lp_attempts"] == 11
        and len(failure["branches"]["fixed"]["records"]) == 10
        and diag["complete"] is diag["passed"] is True
        and diag["new_lp_attempts"] == 2
        and diag["failure_ref"] == FAILURE
        and diag["solves"][0]["maximum_inequality_violation_m"] > 1e-8
        and diag["solves"][1]["maximum_inequality_violation_m"] <= 1e-8,
        "The preserved failure and two-solve tolerance diagnosis are required",
    )
    previous.CONFIG, previous.PLAN, previous.OUTPUT, previous.QA, previous.FIXED = CONFIG, PLAN, OUTPUT, QA, FIXED
    previous.solve_infinity, previous.publish_result = solve_precise, recovered_publication
    return previous


def main():
    bind().main()


if __name__ == "__main__":
    main()
