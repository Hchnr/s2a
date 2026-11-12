"""Record solver output and constraint residuals for one failed box pair."""

from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

from s2a.infrastructure.publication import publish_result
from s2a.research.native_runtime.bilateral_box_feasibility import pair_constraints
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_bowl_fixed_h40 as physical
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/gripper/bowl_static_close_lp_diagnostic_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/gripper/bowl_static_close_lp_diagnostic_20261112.md"
OUTPUT = ROOT / "outputs/bowl_static_close_lp_diagnostic_20261112/result.json"
QA = ROOT / "outputs/qa/bowl_static_close_lp_diagnostic_20261112/result.json"
FAILURE = {
    "path": str(ROOT / "outputs/bowl_static_close_feasibility_20261112/result.json"),
    "sha256": "cba6b1b59e6b7d2c237705281ad0ca9863d09729080a843815262fb74214cdd2",
}
GEOMETRY = {
    "path": str(ROOT / "outputs/bowl_pad_alignment_recovery_audit_20261112/result.json"),
    "sha256": "6fb6ad52ce9a81f0e241215578f151b368235813db480e4a892819c8a9175447",
}
PHYSICAL = {
    "path": str(ROOT / "outputs/reset_bowl_fixed_h40_audit_20261112/result.json"),
    "sha256": "df5e3d2a7ff15fb432f22aad566e4b35c5ea092e1568b3552c05f52484442e8b",
}
FIXED = dict(
    schema="s2a.bowl-static-close-lp-diagnostic.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    seed=42,
    branch="fixed",
    endpoint_step=48,
    left_target="akita_black_bowl_1_g1",
    right_target="akita_black_bowl_1_g11",
    strict_overlap_margin_m=1e-5,
    solver_tolerances=["default", 1e-9],
    max_new_lp=2,
    previous_lp_attempts=11,
    failure_sha256=FAILURE["sha256"],
    geometry_audit_sha256=GEOMETRY["sha256"],
    fixed_physical_audit_sha256=PHYSICAL["sha256"],
    new_models=0,
    new_forwards=0,
    new_physics_steps=0,
    new_contexts=0,
    new_model_requests=0,
    fresh_cases=0,
    automatic_retry=False,
)


def linear_problem(left, right, error):
    """Reconstruct the original LP, including all six infinity-norm constraints."""
    e = np.asarray(error, dtype=float)
    if e.shape != (3,) or not np.isfinite(e).all():
        raise ValueError("A finite world-frame reference error is required")
    rows = np.vstack((left[0], right[0]))
    a = np.column_stack((rows, np.zeros(len(rows))))
    pose = np.zeros((6, 5))
    for index in range(3):
        pose[2 * index, index], pose[2 * index, 4] = 1, -1
        pose[2 * index + 1, index], pose[2 * index + 1, 4] = -1, -1
    return np.vstack((a, pose)), np.r_[left[1], right[1], np.column_stack((e, -e)).ravel()]


def solve_record(a, b, maximum_close, tolerance):
    """Capture the raw solution and all residuals without imposing a new acceptance rule."""
    options = (
        None
        if tolerance == "default"
        else dict(primal_feasibility_tolerance=tolerance, dual_feasibility_tolerance=tolerance)
    )
    result = linprog(
        [0, 0, 0, 0, 1],
        A_ub=a,
        b_ub=b,
        bounds=[(None, None)] * 3 + [(0, maximum_close), (0, None)],
        method="highs",
        options=options,
    )
    row = dict(
        tolerance=tolerance,
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        iterations=int(result.nit),
        x=None,
        residuals=None,
        maximum_inequality_violation_m=None,
        maximum_bound_violation_m=None,
    )
    if result.x is not None:
        x = np.asarray(result.x, dtype=float)
        prefix.require(x.shape == (5,) and np.isfinite(x).all(), "A returned solution must be finite")
        residual = a @ x - b
        row.update(
            x=x,
            residuals=residual,
            maximum_inequality_violation_m=max(0.0, float(np.max(residual))),
            maximum_bound_violation_m=max(0.0, float(-x[3]), float(x[3] - maximum_close), float(-x[4])),
        )
    return row


def main():
    prefix.environment()
    config, qa = prefix.read(prefix.reference(CONFIG)), prefix.read(prefix.reference(QA))
    prefix.require(config == FIXED, "The exact two-solve diagnostic must be registered")
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"],
        "A complete source-matched 32-worker QA is required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "The QA source snapshot differs")
    failure, geometry, physical_audit = (prefix.read(ref) for ref in (FAILURE, GEOMETRY, PHYSICAL))
    failed = failure["branches"]["fixed"]
    prefix.require(
        failure["complete"] is failure["passed"] is False
        and failure["error"]["message"] == "A finite, constraint-verified linear solution is required"
        and failed["lp_attempts"] == 11
        and len(failed["records"]) == 10
        and failed["refinements"] == 0
        and not failed["witnesses"]
        and geometry["complete"]
        is geometry["passed"]
        is physical_audit["complete"]
        is physical_audit["passed"]
        is True,
        "The original partial failure and prior audited evidence are required",
    )
    worker, branch = prefix.read(geometry["worker_ref"]), prefix.read(physical_audit["producer_branch_ref"])
    first, rest = physical.flatten_fixed(branch)
    period = (first + rest)[-1]
    point = worker["geometry"]["points"]["fixed"][-1]
    prefix.require(
        period["step"] == 47 and point["state_sha256"] == period["endpoint_receipt"]["sim_state_sha256"],
        "The registered physical endpoint must match its saved geometry",
    )
    box = point["pad_alignment"]["box_projections"]
    by_name = {row["name"]: row for row in box}
    axis = point["pad_alignment"]["jaw_axis_world"]
    left = pair_constraints(box[0], by_name[config["left_target"]], axis, 1, config["strict_overlap_margin_m"])
    right = pair_constraints(box[1], by_name[config["right_target"]], axis, -1, config["strict_overlap_margin_m"])
    q = np.asarray(period["completed_feedback"]["score"]["qpos"], dtype=float)
    maximum_close = float(min(q[0], -q[1]))
    a, b = linear_problem(left, right, period["tcp_error"][:3])
    output = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        failure_ref=FAILURE,
        geometry_ref=GEOMETRY,
        physical_ref=PHYSICAL,
        config_ref=prefix.reference(CONFIG),
        plan_ref=prefix.reference(PLAN),
        qa_ref=prefix.reference(QA),
        source_ref=prefix.reference(Path(__file__).resolve()),
        scipy_version=scipy.__version__,
        max_closing_m=maximum_close,
        joint_qpos=q,
        constraint_matrix=a,
        constraint_bound=b,
        new_lp_attempts=0,
        solves=[],
        new_models=0,
        new_forwards=0,
        new_physics_steps=0,
        new_contexts=0,
        new_model_requests=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        for tolerance in config["solver_tolerances"]:
            prefix.require(output["new_lp_attempts"] < config["max_new_lp"], "The two-solve budget is exhausted")
            output["new_lp_attempts"] += 1
            output["solves"].append(solve_record(a, b, maximum_close, tolerance))
        prefix.require(snapshot(ROOT)["sources"] == before, "Sources changed during the LP diagnosis")
        output.update(complete=True, passed=True)
    except BaseException as error:
        output["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish_result(OUTPUT, output), flush=True)


if __name__ == "__main__":
    main()
