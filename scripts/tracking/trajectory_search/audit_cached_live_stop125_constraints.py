"""Inspect the quality solver's existing cache at the audited live stop."""

import argparse
import inspect
import os
import signal
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from s2a.references.generated_twist import reference_world_twist
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import cached_live_stop125_admission as admission
from scripts.tracking.trajectory_search import run_cached_live_stop125_feasibility as base

ROOT = admission.ROOT
CONFIG = ROOT / "configs/tracking/joint_control/cached_live_stop125_constraint_audit_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/cached_live_stop125_constraint_audit_20261111.md"
OUTPUT = ROOT / "outputs/cached_live_stop125_constraint_audit_20261111"
QA = ROOT / "outputs/qa/cached_live_stop125_constraint_audit_20261111/result.json"
DRY = ROOT / "outputs/cached_live_stop125_constraint_audit_dry_20261111/result.json"
MODULE = "scripts.tracking.trajectory_search.audit_cached_live_stop125_constraints"
FIXED = {
    "schema": "s2a.cached-live-stop125-constraint-audit.v1",
    "output": "outputs/cached_live_stop125_constraint_audit_20261111",
    "prior_result": "outputs/cached_live_stop125_feasibility_20261111/worker/result.json",
    "prior_result_sha256": "33ca81bddb311de56a1d46f7d37eef564e217e880c149b996052af0aa0fabdb1",
    "case_index": 8,
    "branch": "quality_past",
    "max_models": 1,
    "max_forward_queries": 1,
    "max_optimizer_calls": 1,
    "max_optimizer_responses": 160,
    "extra_model_responses": 0,
    "independent_witness_checks": 0,
    "physics_steps": 0,
    "world_model_calls": 0,
    "fresh_cases": 0,
    "worker_timeout_seconds": 600,
    "automatic_retry": False,
}


def source_refs():
    return base.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "tests/research/native_runtime/test_cached_live_stop125_constraint_audit.py",
        )
    ]


def load_inputs():
    inputs = admission.load_inputs()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact cache-audit budget is required",
    )
    prior_ref = {"path": str(ROOT / config["prior_result"]), "sha256": config["prior_result_sha256"]}
    prior = prefix.read(prior_ref)
    expected = prior["analysis"]["candidates"]["quality_past"]
    prefix.require(
        prior["complete"] is prior["passed"] is True
        and expected["status"] == "nonlinear_search_no_witness"
        and expected["optimizer_success"] is True
        and expected["witness_found"] is False
        and expected["response_evaluations"] == 58,
        "The original completed quality search is required",
    )
    inputs["source_sha256"].update({config_ref["path"]: config_ref["sha256"], prior_ref["path"]: prior_ref["sha256"]})
    inputs.update(config=config, audit_config_ref=config_ref, prior_ref=prior_ref, prior=prior)
    return inputs


def input_receipt(inputs):
    return {
        "inputs": admission.input_receipt(inputs),
        "audit_config_ref": inputs["audit_config_ref"],
        "prior_ref": inputs["prior_ref"],
    }


def inspect_cache(cache, bounds, constraints, previous):
    """Evaluate the original postfilter over cached scores without model calls."""
    prefix.require(len(constraints) in (3, 4), "Original rate, joint, pose and optional terminal constraints")
    rate, joint = constraints[:2]
    prefix.require(isinstance(constraints[2], dict), "Original first-pose constraint")
    terminal = constraints[3] if len(constraints) == 4 else None
    rows = []
    for key, item in cache.items():
        x = np.asarray(key, dtype=float)
        center = np.r_[previous, np.zeros(len(x) - len(previous))]
        rate_abs = float(np.max(np.abs(rate.A @ x - center)))
        joint_value = joint.A @ x
        joint_slack = float(np.min(np.r_[joint_value - joint.lb, joint.ub - joint_value]))
        box_slack = float(np.min(np.r_[x - bounds.lb, bounds.ub - x]))
        terminal_excess = None if terminal is None else float(np.max(terminal.A @ x - terminal.ub))
        error = np.asarray(item["score"]["residual"][:6]) / np.array([100.0] * 3 + [20.0] * 3)
        position, rotation = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
        checks = {
            "box": bool(np.all(x >= bounds.lb - 1e-12) and np.all(x <= bounds.ub + 1e-12)),
            "rate": rate_abs <= 0.25 + 1e-12,
            "joint_braking": bool(np.all(joint_value >= joint.lb - 1e-10) and np.all(joint_value <= joint.ub + 1e-10)),
            "terminal": terminal_excess is None or terminal_excess <= 1e-10,
            "position": position <= 0.01 + 1e-10,
            "rotation": rotation <= 0.05 + 1e-10,
        }
        rows.append(
            {
                "commands": x.reshape(-1, 6),
                "objective": float(item["score"]["objective"]),
                "position_error_m": position,
                "rotation_error_rad": rotation,
                "box_minimum_slack": box_slack,
                "maximum_rate": rate_abs,
                "joint_braking_minimum_slack_rad": joint_slack,
                "terminal_maximum_excess": terminal_excess,
                "checks": checks,
                "accepted": all(checks.values()),
                "failed_checks": [name for name, passed in checks.items() if not passed],
            }
        )
    return rows


def isolated(function, replacements):
    namespace = function.__globals__.copy()
    namespace.update(replacements)
    clone = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return clone


def single_search(context, stopped, targets, anchor, stats, *, quality_solver, joint_solver, auditor, results):
    del joint_solver, auditor
    args = (
        context["model"],
        context["velocity"],
        stopped["shifted_seed"],
        stopped["previous"][:6],
        context["interface"],
        context["site"],
        context["site_from_tcp"],
        targets[5:13],
        np.asarray([reference_world_twist(targets, index, anchor, 0.05)["twist"] for index in range(5, 13)]),
        context["limits"][:, 0],
        context["limits"][:, 1],
    )
    captured = {}
    original_minimize = quality_solver.__globals__["minimize"]

    def observed_minimize(objective, *positional, **kwargs):
        result = original_minimize(objective, *positional, **kwargs)
        evaluate = inspect.getclosurevars(objective).nonlocals["evaluate"]
        cache = inspect.getclosurevars(evaluate).nonlocals["cache"]
        before = stats["optimizer_responses_completed"]
        rows = inspect_cache(cache, kwargs["bounds"], kwargs["constraints"], args[3])
        prefix.require(stats["optimizer_responses_completed"] == before, "Cache inspection must add no model responses")
        final_key = tuple(result.x)
        keys = list(cache)
        captured.update(
            candidates=rows,
            cache_count=len(cache),
            extra_model_responses=0,
            optimizer_final_candidate_index=keys.index(final_key) if final_key in cache else None,
        )
        return result

    solver = isolated(quality_solver, {"minimize": observed_minimize})
    stats["optimizer_calls_attempted"] += 1
    solved = solver(*args, past_inner_error_rad=stopped["past_inner_error_rad"])
    stats["optimizer_calls_completed"] += 1
    prefix.require(
        captured.get("cache_count") == solved["response_evaluations"] == stats["optimizer_responses_completed"] <= 160,
        "The complete original solver cache and model-call count must agree",
    )
    accepted = sum(row["accepted"] for row in captured["candidates"])
    prefix.require(bool(accepted) is solved["witness_found"], "Cached strict filtering must match the original solver")
    results["quality_past"] = {key: value for key, value in solved.items() if key != "rollout"}
    results["cache_audit"] = captured
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
        analyze = isolated(base.analyze, {"analyze_candidates": single_search})
        analyze(inputs, stats, result["analysis"])
        prior = inputs["prior"]["analysis"]
        actual = result["analysis"]
        old, new = prior["candidates"]["quality_past"], actual["candidates"]["quality_past"]
        fields = (
            "status",
            "witness_found",
            "optimizer_success",
            "iterations",
            "response_evaluations",
            "optimizer_message",
        )
        prefix.require(all(old[key] == new[key] for key in fields), "Quality search must match the previous result")
        for key in ("seed_commands", "past_inner_error_rad", "first_pose_caps", "terminal_normals"):
            prior_values, current_values = np.asarray(old[key]), np.asarray(new[key])
            if not prior_values.size and not current_values.size:
                continue  # JSON represents an empty (0, 7) array as [].
            np.testing.assert_array_equal(prior_values, current_values)
        np.testing.assert_allclose(actual["current_tcp_error"], prior["current_tcp_error"], atol=0, rtol=0)
        prefix.require(
            stats["model_initializations_attempted"] == stats["model_initializations_completed"] == 1
            and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == 1
            and stats["optimizer_calls_attempted"] == stats["optimizer_calls_completed"] == 1
            and stats["optimizer_responses_attempted"] == stats["optimizer_responses_completed"] <= 160
            and stats["physics_steps_attempted"] == stats["independent_endpoint_queries_charged"] == 0
            and actual["candidates"]["cache_audit"]["extra_model_responses"] == 0,
            "The registered single-search static budget must hold",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(registration["source_refs"] == source_refs(), "Sources changed during cache inspection")
        result.update(complete=True, passed=True, previous_search_matched=True)
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
    check = isolated(base.check_admission, {"QA": QA, "DRY": DRY, "admission": binding, "source_refs": source_refs})
    owner = isolated(
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
