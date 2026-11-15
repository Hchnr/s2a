"""Search at step121 using only the preceding completed physical feedback."""

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite, pose_error, validate_transform
from s2a.infrastructure.publication import publish_result
from s2a.references.action_reference import decode_action_targets
from s2a.references.generated_twist import reference_world_twist
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.trajectory_search import quality_capped_inner_joint_plan as quality
from s2a.research.trajectory_search import staged_first_feasibility as staged
from s2a.research.trajectory_search.causal_first_pose_constraint import CausalPoseConstraint
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as physical
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import early_causal_innovation_admission as admission
from scripts.tracking.trajectory_search import run_cached_live_stop125_feasibility as base
from scripts.tracking.trajectory_search import run_early_causal_innovation_quality as original
from scripts.verification.reference_ports import audit_causal_stream_live_cached as live_audit
from scripts.wam.tracking.preview import causal_stream_live_cached_admission as live

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/anticipatory_causal_pose_static_20261113/result.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/anticipatory_causal_pose_20261113.md"
QA = ROOT / "outputs/qa/anticipatory_causal_pose_static_20261113/result.json"
LIVE_AUDIT = {
    "path": str(ROOT / "outputs/causal_stream_live_cached_audit_20261111/result.json"),
    "sha256": "65f51ff8b9135ba01462ec96f662ed648024eb1df5e8dee1b3fef09dbaaa87c9",
}


def completed_step120(producer, checker):
    """Extract only step120 feedback, with no task labels or later outcomes."""
    left = [row for row in physical.periods(producer) if row["step"] == 120]
    right = [row for row in physical.periods(checker) if row["step"] == 120]
    if len(left) != 1 or len(right) != 1:
        raise ValueError("One completed producer and checker step120 are required")
    actual = left[0]
    physical.matched_prior_period(actual, right[0])
    state, previous = finite(actual["actual_state"], (79,)).copy(), finite(actual["command"], (7,)).copy()
    if state_sha256(state) != actual["endpoint_receipt"]["sim_state_sha256"]:
        raise ValueError("The step120 endpoint must match the measured state")
    quality = actual["decision"]["quality"]
    history = actual["completed_feedback"]["past_inner_error"]
    if not (
        quality["verification"]["passed"] is True
        and history["available"] is True
        and history["last_completed_step"] == 120
        and history["completed_periods"] == 1
        and history["total_completed_periods"] == 41
        and history["future_guarantee"] is False
    ):
        raise ValueError("Only the current completed and verified step120 history is admissible")
    plan = finite(actual["decision"]["refined_plan"], (8, 6)).copy()
    np.testing.assert_array_equal(plan[0], previous[:6])
    seed = np.vstack((plan[1:], plan[-1]))
    if (
        np.max(np.abs(seed)) > 1 + 1e-12
        or np.max(np.abs(np.diff(np.vstack((previous[:6], seed)), axis=0))) > 0.25 + 1e-12
    ):
        raise ValueError("The completed shifted seed must preserve the original command box and rate")
    predicted, measured = finite(quality["first_tcp_error"], (6,)), finite(actual["tcp_error"], (6,))
    discrepancy = Rotation.from_rotvec(measured[3:]).inv() * Rotation.from_rotvec(predicted[3:])
    profile = finite(history["inner_position_error_rad"], (25, 7)).copy()
    if np.any(profile < 0) or np.any(np.diff(profile, axis=0) < 0):
        raise ValueError("The past inner-error profile must remain nonnegative and monotonic")
    return {
        "state": state,
        "previous": previous,
        "shifted_seed": seed,
        "past_inner_error_rad": profile,
        "translation_innovation": measured[:3] - predicted[:3],
        "rotation_discrepancy": discrepancy.as_matrix(),
    }


def load_inputs():
    """Bind only the step120 reference and completed action to the search."""
    cached = live.load_inputs()
    consumed = dict(cached["source_sha256"])
    audit = prefix.read(LIVE_AUDIT, consumed)
    if not (audit["complete"] is audit["passed"] is audit["checker_policy_isolated"] is True):
        raise ValueError("The saved current-image reference must have an independent replay audit")
    owner_ref = {"path": str(live_audit.RUN / "producer/result.json"), "sha256": live_audit.PRODUCER_SHA}
    owner = prefix.read(owner_ref, consumed)
    live_producer = prefix.read(owner["branch_ref"], consumed)
    runtime = live_producer["runtime"]
    captures = [row for row in runtime["streaming_captures"] if row["frame_index"] == 120]
    exchanges = [row for row in runtime["streaming_model_exchanges"] if row["step"] == 120]
    if len(captures) != 1 or len(exchanges) != 1:
        raise ValueError("Exactly one actual step120 image and response are required")
    capture, exchange = captures[0], exchanges[0]
    response = prefix.read(exchange["response_ref"], consumed)
    prefix.checked(capture["canvas_png"], consumed)
    if response["step"] != 120 or response["image_sha256"] != capture["canvas_png"]["sha256"]:
        raise ValueError("The generated reference must match the actual step120 image")
    old = cached["old"]["old"]["base"]["previous"]
    tool = validate_transform(np.asarray(old["tool"], dtype=float))
    anchor = validate_transform(np.asarray(capture["current_source_pose"], dtype=float)) @ tool
    targets, _openings = decode_action_targets(anchor, response["action"], horizon=40)
    producer = prefix.read(admission.PRODUCER, consumed)
    checker = prefix.read(admission.CHECKER, consumed)
    if not all(item["complete"] is item["passed"] is True for item in (producer, checker)):
        raise ValueError("Both original physical phases must have completed")
    return {
        "stop_inputs": completed_step120(producer, checker),
        "record_xml": old["record_xml"],
        "controller": old["controller"],
        "initial_joint": np.asarray(runtime["runtime_initial_joint"], dtype=float),
        "tool": tool,
        "anchor": anchor,
        "targets": targets,
        "source_sha256": consumed,
        "input_receipt": {
            "audit_ref": LIVE_AUDIT,
            "producer_ref": admission.PRODUCER,
            "checker_ref": admission.CHECKER,
            "live_owner_ref": owner_ref,
            "live_producer_ref": owner["branch_ref"],
            "image_ref": capture["canvas_png"],
            "response_ref": exchange["response_ref"],
        },
    }


def run_branch(context, stopped, targets, anchor, stats, *, raw, innovation, auditor, results):
    """Select at step121; constrain and independently audit step122 as point two."""
    del raw, innovation, auditor
    previous, tool = stopped["previous"][:6], context["site_from_tcp"]
    desired = targets[1:9]
    twists = np.asarray([reference_world_twist(targets, index, anchor, 0.05)["twist"] for index in range(1, 9)])
    constraint = CausalPoseConstraint(
        stopped["translation_innovation"],
        stopped["rotation_discrepancy"],
        pose_index=1,
        position_cap_m=0.007,
        rotation_cap_rad=0.04,
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
    current_error = pose_error(desired[0], context["site"] @ tool)
    stats["optimizer_calls_attempted"] += 1
    before = stats["optimizer_responses_completed"]
    solved = staged.solve_staged_first_feasibility(
        *args, past_inner_error_rad=stopped["past_inner_error_rad"], first_pose_constraint=constraint
    )
    stats["optimizer_calls_completed"] += 1
    if not 0 <= solved["response_evaluations"] <= 320 or solved["response_evaluations"] != (
        stats["optimizer_responses_completed"] - before
    ):
        raise ValueError("Actual model responses must match the shared 320-response limit")
    candidate = solved.get("best_joint_safe")
    row = {key: value for key, value in solved.items() if key != "best_joint_safe"}
    row.update(witness_found=candidate is not None, certified_model_witness=False, independent_verification=None)
    results.update(anticipatory=row, current_tcp_error=current_error)
    if candidate is None:
        return results
    row.update({key: value for key, value in candidate.items() if key != "rollout"})
    stats["independent_endpoint_queries_charged"] += 200
    stats["independent_transition_steps_charged"] += 2600
    checked = audit_full_horizon_witness(
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
        desired,
        context["limits"][:, 0],
        context["limits"][:, 1],
        past_inner_error_rad=stopped["past_inner_error_rad"],
        enforce_pose_caps=False,
    )
    errors = finite(checked["tcp_errors"], (8, 6))
    if not (
        checked["passed"] is checked["braking_constraint_enforced"] is True
        and checked["independent_endpoint_queries"] == 200
        and checked["independent_transition_steps"] == 2600
        and np.all(np.linalg.norm(errors[:2, :3], axis=1) <= 0.01 + 1e-10)
        and np.all(np.linalg.norm(errors[:2, 3:], axis=1) <= 0.05 + 1e-10)
    ):
        raise ValueError("The original safety and both independent raw pose gates must pass")
    discrepancy = Rotation.from_matrix(stopped["rotation_discrepancy"])
    corrected = np.r_[
        errors[1, :3] + stopped["translation_innovation"],
        (Rotation.from_rotvec(errors[1, 3:]) * discrepancy.inv()).as_rotvec(),
    ]
    np.testing.assert_allclose(corrected, candidate["innovation_corrected_tcp_error"], atol=1e-10, rtol=0)
    if np.linalg.norm(corrected[:3]) > 0.007 + 1e-10 or np.linalg.norm(corrected[3:]) > 0.04 + 1e-10:
        raise ValueError("The independently reconstructed second-pose innovation gate must pass")
    row.update(certified_model_witness=True, independent_verification=checked, independent_corrected_error=corrected)
    return results


def execute():
    prefix.environment()
    if os.environ.get("MUJOCO_GL") != "disable":
        raise ValueError("The saved-state preview requires CPU-only MuJoCo without rendering")
    qa = prefix.read(prefix.reference(QA))
    if not (
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and snapshot(ROOT)["sources"] == prefix.read(qa["source_before_ref"])["sources"]
    ):
        raise ValueError("The source-matched CPU regression must pass before the search")
    inputs = load_inputs()
    paths = (
        Path(__file__),
        PLAN,
        QA,
        ROOT / "src/s2a/research/trajectory_search/causal_first_pose_constraint.py",
        ROOT / "src/s2a/research/trajectory_search/staged_first_feasibility.py",
        ROOT / "tests/research/trajectory_search/test_causal_first_pose_constraint.py",
    )
    sources = (
        original.source_refs() + [prefix.reference(path) for path in paths] + [admission.PRODUCER, admission.CHECKER]
    )
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
        input_receipt=inputs["input_receipt"],
        input_sha256=inputs["source_sha256"],
        completed_step=120,
        decision_step=121,
        constrained_future_step=122,
        reference_generated_step=120,
        model_requests=0,
        physics_steps=0,
        fresh_cases=0,
        task_labels_used=False,
        automatic_retry=False,
        stats=stats,
        analysis=analysis,
    )
    old_branch, old_rollout = original.run_branches, staged.rollout_joint_preview
    try:
        original.run_branches = run_branch
        staged.rollout_joint_preview = lambda *args, **kwargs: quality.rollout_joint_preview(*args, **kwargs)
        original.analyze(inputs, stats, analysis)
        current = analysis["candidates"]["current_tcp_error"]
        analysis.update(
            decision_step=121,
            last_completed_step=120,
            generated_step=120,
            reference_cursor=1,
            current_tcp_error=current,
            current_position_error_m=float(np.linalg.norm(current[:3])),
            current_rotation_error_rad=float(np.linalg.norm(current[3:])),
        )
        result["tight_margin_witness"] = analysis["candidates"]["anticipatory"]["certified_model_witness"]
        if not (
            stats["model_initializations_attempted"] == stats["model_initializations_completed"] == 1
            and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == 1
            and stats["optimizer_calls_attempted"] == stats["optimizer_calls_completed"] == 1
            and stats["optimizer_responses_attempted"] == stats["optimizer_responses_completed"] <= 320
            and stats["independent_endpoint_queries_charged"] <= 200
            and stats["independent_transition_steps_charged"] <= 2600
            and stats["physics_steps_attempted"] == 0
        ):
            raise ValueError("The preregistered preview resource budget must hold")
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        if sources != original.source_refs() + [prefix.reference(path) for path in paths] + [
            prefix.reference(Path(admission.PRODUCER["path"])),
            prefix.reference(Path(admission.CHECKER["path"])),
        ]:
            raise ValueError("The preview inputs or sources changed during execution")
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
