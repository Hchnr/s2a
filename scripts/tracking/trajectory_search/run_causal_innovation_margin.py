"""Compare a fixed causal TCP margin with the raw solver at a saved state."""

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.infrastructure.publication import publish_result
from s2a.research.trajectory_search import innovation_capped_quality_solver as solver
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import early_causal_innovation_admission as admission
from scripts.tracking.trajectory_search import run_cached_live_stop125_feasibility as base
from scripts.tracking.trajectory_search import run_early_causal_innovation_quality as original

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/causal_innovation_margin_static_20261113/result.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/causal_innovation_margin_20261113.md"
QA = ROOT / "outputs/qa/causal_innovation_margin_static_20261113/result.json"
POSITION_CAP_M = 0.007
ROTATION_CAP_RAD = 0.04


def check_margin(row, translation, rotation):
    """Require an independently audited, correctly reconstructed tight witness."""
    if row.get("witness_found") is False:
        return False
    if row.get("witness_found") is not True:
        raise ValueError("An explicit witness outcome is required")
    verification = row.get("independent_verification")
    if (
        not isinstance(verification, dict)
        or verification.get("passed") is not True
        or verification.get("first_pose_caps_satisfied") is not True
        or verification.get("first_pose_caps_enforced") is not True
        or row.get("certified_model_witness") is not True
    ):
        raise ValueError("The original independent quality and safety audit must pass")
    raw = np.asarray(verification["first_tcp_error"], dtype=float)
    translation, rotation = np.asarray(translation, dtype=float), np.asarray(rotation, dtype=float)
    if raw.shape != (6,) or translation.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("A six-dimensional error and completed world-frame innovation are required")
    checked = np.r_[
        raw[:3] + translation, (Rotation.from_rotvec(raw[3:]) * Rotation.from_matrix(rotation).inv()).as_rotvec()
    ]
    claimed = np.asarray(row["innovation_corrected_first_tcp_error"], dtype=float)
    if checked.shape != (6,) or not np.isfinite(checked).all():
        raise ValueError("A finite independently calculated corrected TCP error is required")
    np.testing.assert_allclose(checked, row["independent_corrected_error"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(checked, claimed, atol=1e-10, rtol=0)
    if np.linalg.norm(checked[:3]) > POSITION_CAP_M + 1e-10 or np.linalg.norm(checked[3:]) > ROTATION_CAP_RAD + 1e-10:
        raise ValueError("The independent corrected TCP prediction exceeds the fixed margin")
    return True


def execute():
    prefix.environment()
    if os.environ.get("MUJOCO_GL") != "disable":
        raise ValueError("The saved-state search requires a CPU-only, no-render MuJoCo environment")
    qa = prefix.read(prefix.reference(QA))
    if not (
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and snapshot(ROOT)["sources"] == prefix.read(qa["source_before_ref"])["sources"]
    ):
        raise ValueError("The registered CPU regression must pass before the comparison")
    inputs = admission.load_inputs()
    sources = original.source_refs() + [prefix.reference(Path(__file__)), prefix.reference(PLAN), prefix.reference(QA)]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    stats, analysis = base.new_stats(), {}
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        source_commit=commit,
        source_refs=sources,
        input_receipt=admission.input_receipt(inputs),
        input_sha256=inputs["source_sha256"],
        corrected_position_cap_m=POSITION_CAP_M,
        corrected_rotation_cap_rad=ROTATION_CAP_RAD,
        stats=stats,
        analysis=analysis,
        model_requests=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    previous = solver.solve_interior_plan

    def tightened(*args, **kwargs):
        return previous(
            *args,
            corrected_position_cap_m=POSITION_CAP_M,
            corrected_rotation_cap_rad=ROTATION_CAP_RAD,
            **kwargs,
        )

    try:
        solver.solve_interior_plan = tightened
        original.analyze(inputs, stats, analysis)
        stopped = inputs["stop_inputs"]
        result["tight_margin_witness"] = check_margin(
            analysis["candidates"]["innovation_quality"],
            stopped["translation_innovation"],
            stopped["rotation_discrepancy"],
        )
        if not (
            stats["model_initializations_attempted"] == stats["model_initializations_completed"] == 1
            and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == 1
            and stats["optimizer_calls_attempted"] == stats["optimizer_calls_completed"] == 2
            and stats["optimizer_responses_attempted"] == stats["optimizer_responses_completed"] <= 320
            and stats["independent_endpoint_queries_charged"] <= 400
            and stats["independent_transition_steps_charged"] <= 5200
            and stats["physics_steps_attempted"] == 0
        ):
            raise ValueError("The fixed static-comparison budget must hold")
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        if sources != original.source_refs() + [
            prefix.reference(Path(__file__)),
            prefix.reference(PLAN),
            prefix.reference(QA),
        ]:
            raise ValueError("The comparison sources changed during execution")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        solver.solve_interior_plan = previous
        print(publish_result(OUTPUT, result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    execute()


if __name__ == "__main__":
    main()
