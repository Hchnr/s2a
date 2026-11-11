"""Evaluate the opt-in rate-repair solver against the saved live stop."""

import argparse
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from s2a.references.generated_twist import reference_world_twist
from s2a.research.trajectory_search import quality_rate_repair_solver as solver
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_cached_live_stop125_constraints as audit
from scripts.tracking.trajectory_search import cached_live_stop125_admission as admission
from scripts.tracking.trajectory_search import repair_cached_live_stop125_rate as prior_repair
from scripts.tracking.trajectory_search import run_cached_live_stop125_feasibility as base

ROOT = admission.ROOT
CONFIG = ROOT / "configs/tracking/joint_control/quality_rate_repair_solver_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/quality_rate_repair_solver_20261111.md"
OUTPUT = ROOT / "outputs/quality_rate_repair_solver_20261111"
QA = ROOT / "outputs/qa/quality_rate_repair_solver_20261111/result.json"
DRY = ROOT / "outputs/quality_rate_repair_solver_dry_20261111/result.json"
MODULE = "scripts.tracking.trajectory_search.run_quality_rate_repair_solver"
FIXED = {
    "schema": "s2a.quality-rate-repair-solver.v1",
    "output": "outputs/quality_rate_repair_solver_20261111",
    "static_repair": "outputs/cached_live_stop125_rate_repair_20261111/worker/result.json",
    "static_repair_sha256": "dc3b04563eec8cd97f46747507fcc5e609406642e61bf4dada6176453c7cb744",
    "case_index": 8,
    "max_models": 1,
    "max_forward_queries": 1,
    "max_optimizer_calls": 1,
    "max_model_responses": 160,
    "max_independent_endpoint_queries": 200,
    "max_independent_transition_steps": 2600,
    "physics_steps": 0,
    "world_model_calls": 0,
    "fresh_cases": 0,
    "worker_timeout_seconds": 600,
    "automatic_retry": False,
}


def source_refs():
    return prior_repair.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/trajectory_search/quality_rate_repair_solver.py",
            ROOT / "tests/research/trajectory_search/test_quality_rate_repair_solver.py",
            ROOT / "tests/research/native_runtime/test_quality_rate_repair_solver.py",
        )
    ]


def load_inputs():
    inputs = admission.load_inputs()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact online-solver static budget is required",
    )
    prior_ref = {"path": str(ROOT / config["static_repair"]), "sha256": config["static_repair_sha256"]}
    prior = prefix.read(prior_ref)
    prefix.require(
        prior["complete"] is prior["passed"] is True
        and prior["analysis"]["candidates"]["repair"]["certified_model_witness"] is True,
        "The independently checked static repair is required only as a later comparison",
    )
    inputs["source_sha256"].update({config_ref["path"]: config_ref["sha256"], prior_ref["path"]: prior_ref["sha256"]})
    inputs.update(config=config, solver_config_ref=config_ref, prior_ref=prior_ref, prior=prior)
    return inputs


def input_receipt(inputs):
    return {
        "inputs": admission.input_receipt(inputs),
        "solver_config_ref": inputs["solver_config_ref"],
        "prior_ref": inputs["prior_ref"],
    }


def single_search(context, stopped, targets, anchor, stats, *, quality_solver, joint_solver, auditor, results):
    del quality_solver, joint_solver
    previous, tool = stopped["previous"][:6], context["site_from_tcp"]
    desired = targets[5:13]
    twists = np.asarray([reference_world_twist(targets, index, anchor, 0.05)["twist"] for index in range(5, 13)])
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
    solved = solver.solve_interior_plan(*args, past_inner_error_rad=stopped["past_inner_error_rad"])
    stats["optimizer_calls_completed"] += 1
    prefix.require(
        0 <= solved["response_evaluations"] <= 160
        and solved["response_evaluations"] == stats["optimizer_responses_completed"] - before,
        "Original and repair model calls must fit the same registered response budget",
    )
    row = {key: value for key, value in solved.items() if key != "rollout"}
    results["quality_rate_repair"] = row
    row.update(certified_model_witness=False, independent_verification=None)
    if solved["witness_found"] is True:
        stats["independent_endpoint_queries_charged"] += 200
        stats["independent_transition_steps_charged"] += 2600
        try:
            checked = auditor(
                solved,
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
        except (AssertionError, ValueError) as error:
            row["independent_verification"] = {"passed": False, "error": str(error)}
        else:
            prefix.require(
                checked["passed"] is checked["first_pose_caps_satisfied"] is checked["first_pose_caps_enforced"] is True
                and checked["independent_endpoint_queries"] == 200
                and checked["independent_transition_steps"] == 2600,
                "The independently reconstructed quality and safety audit is required",
            )
            row.update(certified_model_witness=True, independent_verification=checked)
    return results


def worker():
    def interrupted(signum, _frame):
        raise InterruptedError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    stats = base.new_stats()
    result = dict(complete=False, passed=False, stats=stats, analysis={}, world_model_calls=0, fresh_cases=0)
    try:
        prefix.environment()
        prefix.require(os.environ.get("MUJOCO_GL") == "disable", "No graphics context may be initialized")
        registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
        inputs = load_inputs()
        prefix.require(
            registration["inputs"] == input_receipt(inputs) and registration["source_refs"] == source_refs(),
            "Worker inputs and sources must match registration",
        )
        analyze = audit.isolated(base.analyze, {"analyze_candidates": single_search})
        analyze(inputs, stats, result["analysis"])
        row = result["analysis"]["candidates"]["quality_rate_repair"]
        if row["witness_found"] is True:
            # The saved candidate is read only after the solver has finished.
            prior = inputs["prior"]["analysis"]["candidates"]["repair"]
            row["comparison_to_previously_saved_repair"] = {
                "max_command_difference": float(np.max(np.abs(row["planned_commands"] - prior["planned_commands"]))),
                "max_first_tcp_error_difference": float(
                    np.max(np.abs(row["first_tcp_error"] - prior["first_tcp_error"]))
                ),
                "used_as_solver_input": False,
            }
        prefix.require(
            stats["model_initializations_attempted"] == stats["model_initializations_completed"] == 1
            and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == 1
            and stats["optimizer_calls_attempted"] == stats["optimizer_calls_completed"] == 1
            and stats["optimizer_responses_attempted"] == stats["optimizer_responses_completed"] <= 160
            and stats["independent_endpoint_queries_charged"] <= 200
            and stats["independent_transition_steps_charged"] <= 2600
            and stats["physics_steps_attempted"] == 0,
            "The registered single-search model budget must hold",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(registration["source_refs"] == source_refs(), "Sources changed during solver evaluation")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / "worker/result.json", result), flush=True)


def dry_run():
    inputs = load_inputs()
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=input_receipt(inputs),
                source_refs=source_refs(),
                model_initializations=0,
                forward_queries=0,
                optimizer_calls=0,
                physics_steps=0,
                world_model_calls=0,
            ),
        ),
        flush=True,
    )


def execute():
    binding = SimpleNamespace(load_inputs=load_inputs, input_receipt=input_receipt)
    check = audit.isolated(
        base.check_admission, {"QA": QA, "DRY": DRY, "admission": binding, "source_refs": source_refs}
    )
    owner = audit.isolated(
        base.execute,
        {
            "OUTPUT": OUTPUT,
            "QA": QA,
            "DRY": DRY,
            "MODULE": MODULE,
            "admission": binding,
            "source_refs": source_refs,
            "check_admission": check,
        },
    )
    owner()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.execute:
        execute()
    else:
        worker()


if __name__ == "__main__":
    main()
