"""Independently check a tiny rate repair of the cached live-stop candidate."""

import argparse
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.geometry import pose_error
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.trajectory_search.rate_boundary_repair import repair_nearby_rate_plan
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_cached_live_stop125_constraints as audit
from scripts.tracking.trajectory_search import cached_live_stop125_admission as admission
from scripts.tracking.trajectory_search import run_cached_live_stop125_feasibility as base

ROOT = admission.ROOT
CONFIG = ROOT / "configs/tracking/joint_control/cached_live_stop125_rate_repair_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/cached_live_stop125_rate_repair_20261111.md"
OUTPUT = ROOT / "outputs/cached_live_stop125_rate_repair_20261111"
QA = ROOT / "outputs/qa/cached_live_stop125_rate_repair_20261111/result.json"
DRY = ROOT / "outputs/cached_live_stop125_rate_repair_dry_20261111/result.json"
MODULE = "scripts.tracking.trajectory_search.repair_cached_live_stop125_rate"
FIXED = {
    "schema": "s2a.cached-live-stop125-rate-repair.v1",
    "output": "outputs/cached_live_stop125_rate_repair_20261111",
    "cache_audit": "outputs/cached_live_stop125_constraint_audit_20261111/worker/result.json",
    "cache_audit_sha256": "30625ec96570a85113597b9122ad25eb4888a81feb7117578660a4315080030b",
    "case_index": 8,
    "candidate": "optimizer_final",
    "rate_limit": 0.25,
    "rate_interior": 1e-10,
    "max_command_change": 1e-8,
    "max_models": 1,
    "max_forward_queries": 1,
    "max_preview_responses": 1,
    "max_independent_endpoint_queries": 200,
    "max_independent_transition_steps": 2600,
    "optimizer_calls": 0,
    "physics_steps": 0,
    "world_model_calls": 0,
    "fresh_cases": 0,
    "worker_timeout_seconds": 600,
    "automatic_retry": False,
}


def source_refs():
    return audit.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/s2a/research/trajectory_search/rate_boundary_repair.py",
            ROOT / "tests/research/trajectory_search/test_rate_boundary_repair.py",
            ROOT / "tests/research/native_runtime/test_cached_live_stop125_rate_repair.py",
        )
    ]


def select_candidate(prior, stopped):
    prefix.require(
        prior["complete"] is prior["passed"] is prior["previous_search_matched"] is True,
        "A completed matching cache audit is required",
    )
    cache = prior["analysis"]["candidates"]["cache_audit"]
    index = cache["optimizer_final_candidate_index"]
    prefix.require(
        cache["cache_count"] == len(cache["candidates"]) == 58 and type(index) is int and index == 57,
        "The registered optimizer final candidate must be present",
    )
    row = cache["candidates"][index]
    prefix.require(
        row["accepted"] is False and row["failed_checks"] == ["rate"],
        "The final cached candidate must fail only the original rate postfilter",
    )
    commands = np.asarray(row["commands"], dtype=float)
    prefix.require(commands.shape == (8, 6) and np.all(np.isfinite(commands)), "Eight finite cached commands")
    actual_rate = np.max(np.abs(np.diff(np.vstack((stopped["previous"][:6], commands)), axis=0)))
    np.testing.assert_allclose(actual_rate, row["maximum_rate"], atol=1e-15, rtol=0)
    prefix.require(0.25 + 1e-12 < actual_rate < 0.25 + 1e-8, "Only the registered tiny rate excess may be repaired")
    primary = prior["analysis"]["candidates"]["quality_past"]
    return {"commands": commands.copy(), "terminal_normals": np.asarray(primary["terminal_normals"]).reshape(-1, 7)}


def load_inputs():
    inputs = admission.load_inputs()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact rate-repair budget is required",
    )
    prior_ref = {"path": str(ROOT / config["cache_audit"]), "sha256": config["cache_audit_sha256"]}
    prior = prefix.read(prior_ref)
    candidate = select_candidate(prior, inputs["stop_inputs"])
    inputs["source_sha256"].update({config_ref["path"]: config_ref["sha256"], prior_ref["path"]: prior_ref["sha256"]})
    inputs.update(config=config, repair_config_ref=config_ref, prior_ref=prior_ref, candidate=candidate)
    return inputs


def input_receipt(inputs):
    return {
        "inputs": admission.input_receipt(inputs),
        "repair_config_ref": inputs["repair_config_ref"],
        "prior_ref": inputs["prior_ref"],
    }


def repair_search(inputs):
    def run(context, stopped, targets, anchor, stats, *, quality_solver, joint_solver, auditor, results):
        del anchor, quality_solver, joint_solver
        interface, previous = context["interface"], stopped["previous"][:6]
        candidate = inputs["candidate"]
        repaired = repair_nearby_rate_plan(
            candidate["commands"],
            previous,
            interface.input_min,
            interface.input_max,
            max_rate=0.25,
            interior=1e-10,
            max_correction=1e-8,
        )
        results["repair"] = repaired
        commands = repaired["planned_commands"]
        deltas = np.asarray([interface.decode(command) for command in commands])
        stats["preview_responses_attempted"] += 1
        prefix.require(stats["preview_responses_attempted"] <= 1, "One repair preview response is allowed")
        rollout = rollout_joint_preview(
            context["model"],
            context["velocity"],
            deltas,
            context["site"],
            context["limits"][:, 0],
            context["limits"][:, 1],
            enforce_limits=False,
        )
        stats["preview_responses_completed"] += 1
        error = pose_error(targets[5], rollout["pose"][0] @ context["site_from_tcp"])
        repaired.update(
            first_tcp_error=error,
            first_position_error_m=float(np.linalg.norm(error[:3])),
            first_rotation_error_rad=float(np.linalg.norm(error[3:])),
            certified_model_witness=False,
        )
        witness = {
            "witness_found": True,
            "planned_commands": commands,
            "rollout": rollout,
            "first_tcp_error": error,
            "terminal_normals": candidate["terminal_normals"],
        }
        stats["independent_endpoint_queries_charged"] += 200
        stats["independent_transition_steps_charged"] += 2600
        try:
            checked = auditor(
                witness,
                context["mass"],
                context["jacobian"],
                context["contract"]["kp"],
                context["contract"]["kd"],
                context["q"],
                context["velocity"],
                context["initial_joint"],
                interface,
                previous,
                context["site"],
                context["site_from_tcp"],
                targets[5],
                context["limits"][:, 0],
                context["limits"][:, 1],
                past_inner_error_rad=stopped["past_inner_error_rad"],
                enforce_pose_caps=True,
            )
        except (AssertionError, ValueError) as error:
            repaired["independent_verification"] = {"passed": False, "error": str(error)}
        else:
            prefix.require(
                checked["passed"] is checked["first_pose_caps_satisfied"] is checked["first_pose_caps_enforced"] is True
                and checked["independent_endpoint_queries"] == 200
                and checked["independent_transition_steps"] == 2600,
                "The complete original quality and safety audit is required",
            )
            repaired.update(independent_verification=checked, certified_model_witness=True)
        return results

    return run


def worker():
    def interrupted(signum, _frame):
        raise InterruptedError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    stats = base.new_stats()
    stats.update(preview_responses_attempted=0, preview_responses_completed=0)
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
        analyze = audit.isolated(base.analyze, {"analyze_candidates": repair_search(inputs)})
        analyze(inputs, stats, result["analysis"])
        prefix.require(
            stats["model_initializations_attempted"] == stats["model_initializations_completed"] == 1
            and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == 1
            and stats["preview_responses_attempted"] == stats["preview_responses_completed"] == 1
            and stats["optimizer_calls_attempted"] == stats["optimizer_responses_attempted"] == 0
            and stats["independent_endpoint_queries_charged"] == 200
            and stats["independent_transition_steps_charged"] == 2600
            and stats["physics_steps_attempted"] == 0,
            "The registered rate-repair static budget must hold",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(registration["source_refs"] == source_refs(), "Sources changed during the repair diagnostic")
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
