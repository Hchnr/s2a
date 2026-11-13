"""Compare a causal first-pose search at the independently replayed state."""

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.references.generated_twist import reference_world_twist
from s2a.research.trajectory_search import quality_capped_inner_joint_plan as quality
from s2a.research.trajectory_search import staged_first_feasibility as staged
from s2a.research.trajectory_search.causal_first_pose_constraint import CausalFirstPoseConstraint
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import early_causal_innovation_admission as admission
from scripts.tracking.trajectory_search import run_cached_live_stop125_feasibility as base
from scripts.tracking.trajectory_search import run_causal_innovation_margin as margin
from scripts.tracking.trajectory_search import run_early_causal_innovation_quality as original

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/causal_first_pose_search_static_20261113/result.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/causal_first_pose_search_20261113.md"
QA = ROOT / "outputs/qa/causal_first_pose_search_static_20261113/result.json"


def run_branch(context, stopped, targets, anchor, stats, *, raw, innovation, auditor, results):
    """Search once with the completed innovation and independently audit a candidate."""
    del raw, innovation
    cursor = 2
    previous, tool = stopped["previous"][:6], context["site_from_tcp"]
    desired = targets[cursor : cursor + 8]
    twists = np.asarray(
        [reference_world_twist(targets, index, anchor, 0.05)["twist"] for index in range(cursor, cursor + 8)]
    )
    constraint = CausalFirstPoseConstraint(
        stopped["translation_innovation"],
        stopped["rotation_discrepancy"],
        position_cap_m=margin.POSITION_CAP_M,
        rotation_cap_rad=margin.ROTATION_CAP_RAD,
    )
    args = (
        context["model"],
        context["velocity"],
        stopped["shifted_seed"],
        previous,
        context["interface"],
        context["site"],
        tool,
        desired,
        twists,
        context["limits"][:, 0],
        context["limits"][:, 1],
    )
    stats["optimizer_calls_attempted"] += 1
    before = stats["optimizer_responses_completed"]
    solved = staged.solve_staged_first_feasibility(
        *args, past_inner_error_rad=stopped["past_inner_error_rad"], first_pose_constraint=constraint
    )
    stats["optimizer_calls_completed"] += 1
    if not 0 <= solved["response_evaluations"] <= 320 or solved["response_evaluations"] != (
        stats["optimizer_responses_completed"] - before
    ):
        raise ValueError("The shared model-response count must match the actual bounded search")
    candidate = solved.get("best_joint_safe")
    row = {key: value for key, value in solved.items() if key != "best_joint_safe"}
    row.update(witness_found=candidate is not None, certified_model_witness=False, independent_verification=None)
    results["staged_causal"] = row
    if candidate is None:
        return results
    row.update({key: value for key, value in candidate.items() if key != "rollout"})
    stats["independent_endpoint_queries_charged"] += 200
    stats["independent_transition_steps_charged"] += 2600
    checked = auditor(
        candidate,
        context["mass"],
        context["jacobian"],
        context["contract"]["kp"],
        context["contract"]["kd"],
        context["q"],
        context["velocity"],
        context["initial_joint"],
        context["interface"],
        previous,
        context["site"],
        tool,
        desired[0],
        context["limits"][:, 0],
        context["limits"][:, 1],
        past_inner_error_rad=stopped["past_inner_error_rad"],
        enforce_pose_caps=True,
    )
    if not (
        checked["passed"] is checked["first_pose_caps_satisfied"] is checked["first_pose_caps_enforced"] is True
        and checked["independent_endpoint_queries"] == 200
        and checked["independent_transition_steps"] == 2600
    ):
        raise ValueError("The original independent first-pose and safety audit must pass")
    row.update(
        certified_model_witness=True,
        independent_verification=checked,
        independent_corrected_error=constraint.correction(checked["first_tcp_error"]),
    )
    margin.check_margin(row, stopped["translation_innovation"], stopped["rotation_discrepancy"])
    return results


def execute():
    prefix.environment()
    if os.environ.get("MUJOCO_GL") != "disable":
        raise ValueError("The saved-state search requires CPU-only MuJoCo without rendering")
    qa = prefix.read(prefix.reference(QA))
    if not (
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and snapshot(ROOT)["sources"] == prefix.read(qa["source_before_ref"])["sources"]
    ):
        raise ValueError("The source-matched CPU regression must pass before the comparison")
    inputs = admission.load_inputs()
    paths = (
        Path(__file__),
        PLAN,
        QA,
        ROOT / "src/s2a/research/trajectory_search/causal_first_pose_constraint.py",
        ROOT / "src/s2a/research/trajectory_search/staged_first_feasibility.py",
        ROOT / "tests/research/trajectory_search/test_causal_first_pose_constraint.py",
    )
    sources = original.source_refs() + [prefix.reference(path) for path in paths]
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
        stats=stats,
        analysis=analysis,
        model_requests=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    old_branch, old_rollout = original.run_branches, staged.rollout_joint_preview
    try:
        original.run_branches = run_branch
        staged.rollout_joint_preview = lambda *args, **kwargs: quality.rollout_joint_preview(*args, **kwargs)
        original.analyze(inputs, stats, analysis)
        result["tight_margin_witness"] = analysis["candidates"]["staged_causal"]["certified_model_witness"]
        if not (
            stats["model_initializations_attempted"] == stats["model_initializations_completed"] == 1
            and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == 1
            and stats["optimizer_calls_attempted"] == stats["optimizer_calls_completed"] == 1
            and stats["optimizer_responses_attempted"] == stats["optimizer_responses_completed"] <= 320
            and stats["independent_endpoint_queries_charged"] <= 200
            and stats["independent_transition_steps_charged"] <= 2600
            and stats["physics_steps_attempted"] == 0
        ):
            raise ValueError("The fixed first-pose comparison budget must hold")
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        if sources != original.source_refs() + [prefix.reference(path) for path in paths]:
            raise ValueError("The comparison sources changed during execution")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        original.run_branches, staged.rollout_joint_preview = old_branch, old_rollout
        print(publish_result(OUTPUT, result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    execute()


if __name__ == "__main__":
    main()
