"""Bound a saved causal-correction candidate in relaxed first-pose domains."""

import argparse
import os
import subprocess
from functools import partial
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import pose_error
from s2a.infrastructure.publication import publish_result
from s2a.research.joint_control.tracking.causal_position_reachability import causal_position_outer_bounds
from s2a.research.joint_control.tracking.stopped_pose_safety_bound import first_pose_linearization, safety_scopes
from s2a.research.response_models.forward.joint_limit_response import predict_joint_response
from s2a.research.trajectory_search import quality_capped_inner_joint_plan as quality
from s2a.research.trajectory_search.feasible_start_restoration import domain_check, linear_domain
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import early_causal_innovation_admission as admission
from scripts.tracking.trajectory_search import run_cached_live_stop125_feasibility as base
from scripts.tracking.trajectory_search import run_early_causal_innovation_quality as original
from scripts.verification.reference_ports.check_stopped_pose_safety_bound import check_support_certificate

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/causal_position_reachability_20261113/result.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/causal_position_reachability_20261113.md"
QA = ROOT / "outputs/qa/causal_position_reachability_20261113/result.json"
PREVIOUS = {
    "path": str(ROOT / "outputs/causal_first_pose_search_static_20261113/result.json"),
    "sha256": "8e6cde0c518f47e720c8e646ff2b8d303680df0b9a22e1373ed6ba19f817977f",
}


def run_bound(context, stopped, targets, _anchor, stats, *, raw, innovation, auditor, results, saved):
    """Rebuild the saved candidate and bound only its corrected position."""
    del raw, innovation, auditor
    previous = stopped["previous"][:6]
    saved_candidate = saved["analysis"]["candidates"]["staged_causal"]["closest_first_rejected"]
    plan = np.asarray(saved_candidate["planned_commands"], dtype=float)
    model, interface, site, tool = (context[key] for key in ("model", "interface", "site", "site_from_tcp"))
    lower, upper = context["limits"][:, 0], context["limits"][:, 1]
    rollout = quality.rollout_joint_preview(
        model,
        context["velocity"],
        np.asarray([interface.decode(command) for command in plan]),
        site,
        lower,
        upper,
        enforce_limits=False,
    )
    domain = linear_domain(
        model,
        plan,
        previous,
        interface,
        lower,
        upper,
        stopped["past_inner_error_rad"],
        rollout,
        terminal_normals=np.empty((0, 7)),
    )
    safety = domain_check(plan, domain, interface)
    if safety["accepted"] is not True:
        raise ValueError("The saved closest plan must belong to the relaxed safety domain")
    scopes = safety_scopes(domain, plan)
    scale = (interface.output_max - interface.output_min) / (interface.input_max - interface.input_min)
    seed_q = rollout["joint_position"][0]
    command_jacobian = rollout["joint_position_jacobian"][0, :, :6] * scale
    linearization = first_pose_linearization(
        site,
        tool,
        context["jacobian"],
        seed_q - context["q"],
        command_jacobian,
        plan[0],
        scopes["box"]["lower"],
        scopes["box"]["upper"],
        targets[2],
    )
    np.testing.assert_allclose(
        linearization["seed_tcp_position"], (rollout["pose"][0] @ tool)[:3, 3], atol=1e-10, rtol=0
    )

    def account(name, event):
        if name != "box":
            stats[f"support_lp_{event}"] += 1
            if stats["support_lp_attempted"] > 2:
                raise ValueError("The two-support-LP budget is exhausted")

    bounds = causal_position_outer_bounds(linearization, stopped["translation_innovation"], scopes, on_support=account)
    stats["independent_first_responses_attempted"] += 1
    stats["independent_transition_steps_charged"] += 25
    response = predict_joint_response(
        model, context["velocity"], interface.decode(plan[0]), lower, upper, substeps=25, enforce_limits=False
    )
    stats["independent_first_responses_completed"] += 1
    np.testing.assert_allclose(response["joint_position"], seed_q, atol=1e-10, rtol=0)
    movement = context["jacobian"] @ (np.asarray(response["joint_position"]) - context["q"])
    predicted_site = site.copy()
    predicted_site[:3, 3] += movement[:3]
    predicted_site[:3, :3] = Rotation.from_rotvec(movement[3:]).as_matrix() @ site[:3, :3]
    raw_error = pose_error(targets[2], predicted_site @ tool)
    np.testing.assert_allclose(raw_error, saved_candidate["first_tcp_error"], atol=1e-10, rtol=0)
    corrected_position = raw_error[:3] + stopped["translation_innovation"]
    np.testing.assert_allclose(corrected_position, bounds["corrected_seed_error_m"], atol=1e-10, rtol=0)
    checks = {}
    for name, scope in scopes.items():
        row = bounds["scopes"][name]
        check = check_support_certificate(row["support_direction"], scope, row["support"])
        if row["position_lower_bound_m"] > np.linalg.norm(corrected_position) + 1e-9:
            raise ValueError("A bound cannot exceed the corrected norm of its feasible seed")
        checks[name] = check
    results["causal_position_bounds"] = {
        "saved_candidate_linear_safety": safety,
        "saved_candidate_first_tcp_error": raw_error,
        "linearization": linearization,
        "bounds": bounds,
        "independent_support_checks": checks,
        "independent_joint_response_max_difference": float(np.max(np.abs(response["joint_position"] - seed_q))),
        "excludes_fixed_0_007_m_cap": {
            name: bool(row["position_lower_bound_m"] > 0.007) for name, row in bounds["scopes"].items()
        },
    }
    return results


def execute():
    prefix.environment()
    if os.environ.get("MUJOCO_GL") != "disable":
        raise ValueError("The saved-state bound requires CPU-only MuJoCo without rendering")
    qa = prefix.read(prefix.reference(QA))
    if not (
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and snapshot(ROOT)["sources"] == prefix.read(qa["source_before_ref"])["sources"]
    ):
        raise ValueError("The source-matched CPU regression must pass before the bound")
    inputs, saved = admission.load_inputs(), prefix.read(PREVIOUS)
    if not (
        saved["complete"] is saved["passed"] is True
        and saved["tight_margin_witness"] is False
        and saved["input_receipt"] == admission.input_receipt(inputs)
        and saved["analysis"]["candidates"]["staged_causal"]["closest_first_rejected"] is not None
    ):
        raise ValueError("The exact failed, independently sourced first-pose search is required")
    paths = (
        Path(__file__),
        PLAN,
        QA,
        ROOT / "src/s2a/research/joint_control/tracking/causal_position_reachability.py",
        ROOT / "tests/research/joint_control/tracking/test_causal_position_reachability.py",
    )
    sources = original.source_refs() + [prefix.reference(path) for path in paths] + [PREVIOUS]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    stats, analysis = base.new_stats(), {}
    stats.update(
        support_lp_attempted=0,
        support_lp_completed=0,
        independent_first_responses_attempted=0,
        independent_first_responses_completed=0,
    )
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
    old_branch = original.run_branches
    try:
        original.run_branches = partial(run_bound, saved=saved)
        original.analyze(inputs, stats, analysis)
        if not (
            stats["model_initializations_attempted"] == stats["model_initializations_completed"] == 1
            and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == 1
            and stats["optimizer_calls_attempted"] == stats["optimizer_calls_completed"] == 0
            and stats["optimizer_responses_attempted"] == stats["optimizer_responses_completed"] == 1
            and stats["support_lp_attempted"] == stats["support_lp_completed"] == 2
            and stats["independent_first_responses_attempted"] == stats["independent_first_responses_completed"] == 1
            and stats["independent_transition_steps_charged"] == 25
            and stats["physics_steps_attempted"] == 0
        ):
            raise ValueError("The fixed position-bound resource budget must hold")
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        if sources != original.source_refs() + [prefix.reference(path) for path in paths] + [
            prefix.reference(Path(PREVIOUS["path"]))
        ]:
            raise ValueError("The bound inputs or sources changed during execution")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        original.run_branches = old_branch
        print(publish_result(OUTPUT, result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    execute()


if __name__ == "__main__":
    main()
