"""Check a past-innovation quality search at the audited step125 stop."""

import argparse
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import finite
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.generated_twist import reference_world_twist
from s2a.research.trajectory_search import innovation_capped_quality_solver as solver
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_cached_live_stop125_constraints as audit
from scripts.tracking.trajectory_search import cached_live_stop125_admission as admission
from scripts.tracking.trajectory_search import run_cached_live_stop125_feasibility as base

ROOT = admission.ROOT
CONFIG = ROOT / "configs/tracking/joint_control/causal_innovation_quality_search_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/causal_innovation_quality_search_20261111.md"
OUTPUT = ROOT / "outputs/causal_innovation_quality_search_20261111"
QA = ROOT / "outputs/qa/causal_innovation_quality_search_20261111/result.json"
DRY = ROOT / "outputs/causal_innovation_quality_search_dry_20261111/result.json"
MODULE = "scripts.tracking.trajectory_search.run_causal_innovation_quality_search"
FIXED = {
    "schema": "s2a.causal-innovation-quality-search.v1",
    "output": "outputs/causal_innovation_quality_search_20261111",
    "innovation_audit": "outputs/causal_tcp_prediction_audit_20261111/result.json",
    "innovation_audit_sha256": "0a70c6505f69ba9e666b13ccb000e8eac2f418b995c515d4db9fcec5efd02c2c",
    "raw_quality": "outputs/quality_rate_repair_solver_20261111/worker/result.json",
    "raw_quality_sha256": "fb9824cba9ce18fc4acb52bcb25c336d8fa173522c3d27be66bc4acbfe7d9a26",
    "case_index": 8,
    "stop_step": 125,
    "last_completed_step": 124,
    "max_models": 1,
    "max_forward_queries": 1,
    "max_optimizer_calls": 1,
    "max_model_responses": 160,
    "max_repair_responses": 1,
    "max_independent_endpoint_queries": 200,
    "max_independent_transition_steps": 2600,
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
            ROOT / "src/s2a/research/trajectory_search/innovation_capped_quality_solver.py",
            ROOT / "src/s2a/research/trajectory_search/quality_rate_repair_solver.py",
            ROOT / "src/s2a/research/trajectory_search/rate_boundary_repair.py",
            ROOT / "tests/research/trajectory_search/test_innovation_capped_quality_solver.py",
            ROOT / "tests/research/native_runtime/test_causal_innovation_quality_search.py",
        )
    ]


def completed_innovation(saved):
    """Recover only the immediately preceding completed period's residual."""
    prefix.require(saved["complete"] is saved["passed"] is True, "A completed causal audit is required")
    previous = [row for row in saved["periods"] if row["step"] == 124]
    prefix.require(len(previous) == 1, "The unique completed step124 is required")
    row = previous[0]
    prefix.require(
        row["completed_innovation"]["step"] == 124 and row["completed_innovation"]["completed_history"] == 45,
        "The innovation must have been completed before step125",
    )
    predicted, actual = finite(row["raw"]["predicted_error"], (6,)), finite(row["actual_error"], (6,))
    translation = actual[:3] - predicted[:3]
    discrepancy = Rotation.from_rotvec(actual[3:]).inv() * Rotation.from_rotvec(predicted[3:])
    np.testing.assert_allclose(translation, row["completed_innovation"]["position_innovation_m"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        discrepancy.as_rotvec(), row["completed_innovation"]["rotation_innovation_rad"], atol=1e-12, rtol=0
    )
    return {"last_completed_step": 124, "translation_m": translation, "rotation_matrix": discrepancy.as_matrix()}


def load_inputs():
    inputs = admission.load_inputs()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact innovation-quality search budget is required",
    )
    refs = {
        key: {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]}
        for key in ("innovation_audit", "raw_quality")
    }
    innovation_audit, raw_quality = (prefix.read(refs[key]) for key in ("innovation_audit", "raw_quality"))
    innovation = completed_innovation(innovation_audit)
    prefix.require(
        raw_quality["complete"] is raw_quality["passed"] is True
        and raw_quality["analysis"]["candidates"]["quality_rate_repair"]["certified_model_witness"] is True,
        "A previous independently checked raw-quality comparison is required",
    )
    inputs["source_sha256"].update({config_ref["path"]: config_ref["sha256"]})
    inputs["source_sha256"].update({ref["path"]: ref["sha256"] for ref in refs.values()})
    inputs.update(
        config=config, search_config_ref=config_ref, search_refs=refs, innovation=innovation, raw_quality=raw_quality
    )
    return inputs


def input_receipt(inputs):
    return {
        "inputs": admission.input_receipt(inputs),
        "search_config_ref": inputs["search_config_ref"],
        "search_refs": inputs["search_refs"],
    }


def single_search(innovation):
    def run(context, stopped, targets, anchor, stats, *, quality_solver, joint_solver, auditor, results):
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
        solved = solver.solve_interior_plan(
            *args,
            translation_innovation=innovation["translation_m"],
            rotation_discrepancy=innovation["rotation_matrix"],
            past_inner_error_rad=stopped["past_inner_error_rad"],
        )
        stats["optimizer_calls_completed"] += 1
        prefix.require(
            0 <= solved["response_evaluations"] <= 160
            and solved["response_evaluations"] == stats["optimizer_responses_completed"] - before,
            "Original and optional repair calls must fit the same 160-response budget",
        )
        row = {key: value for key, value in solved.items() if key != "rollout"}
        results["innovation_quality"] = row
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
                corrected = solver.corrected_error(
                    checked["first_tcp_error"], innovation["translation_m"], innovation["rotation_matrix"]
                )
                np.testing.assert_allclose(
                    corrected, solved["innovation_corrected_first_tcp_error"], atol=1e-10, rtol=0
                )
                prefix.require(
                    checked["passed"]
                    is checked["first_pose_caps_satisfied"]
                    is checked["first_pose_caps_enforced"]
                    is True
                    and checked["independent_endpoint_queries"] == 200
                    and checked["independent_transition_steps"] == 2600
                    and np.linalg.norm(corrected[:3]) <= 0.01 + 1e-10
                    and np.linalg.norm(corrected[3:]) <= 0.05 + 1e-10,
                    "Both original and innovation-corrected independent gates are required",
                )
            except (AssertionError, ValueError) as error:
                row["independent_verification"] = {"passed": False, "error": str(error)}
            else:
                row.update(
                    certified_model_witness=True,
                    independent_verification=checked,
                    independent_corrected_error=corrected,
                )
        return results

    return run


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
        analyze = audit.isolated(base.analyze, {"analyze_candidates": single_search(inputs["innovation"])})
        analyze(inputs, stats, result["analysis"])
        # The saved raw-quality candidate is used for comparison only after search.
        previous = inputs["raw_quality"]["analysis"]["candidates"]["quality_rate_repair"]
        original_error = finite(previous["first_tcp_error"], (6,))
        result["raw_quality_comparison"] = {
            "first_tcp_error": original_error,
            "innovation_corrected_first_tcp_error": solver.corrected_error(
                original_error, inputs["innovation"]["translation_m"], inputs["innovation"]["rotation_matrix"]
            ),
            "used_as_search_input": False,
        }
        prefix.require(
            stats["model_initializations_attempted"] == stats["model_initializations_completed"] == 1
            and stats["forward_queries_attempted"] == stats["forward_queries_completed"] == 1
            and stats["optimizer_calls_attempted"] == stats["optimizer_calls_completed"] == 1
            and stats["optimizer_responses_attempted"] == stats["optimizer_responses_completed"] <= 160
            and stats["independent_endpoint_queries_charged"] <= 200
            and stats["independent_transition_steps_charged"] <= 2600
            and stats["physics_steps_attempted"] == 0,
            "The registered single-search static budget must hold",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(registration["source_refs"] == source_refs(), "Sources changed during innovation search")
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
