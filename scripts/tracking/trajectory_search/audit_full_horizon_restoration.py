"""Restore full-horizon TCP feasibility at earlier causal warnings."""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

from s2a.geometry import pose_error
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.trajectory_search.full_horizon_quality import solve_full_horizon
from s2a.research.trajectory_search.past_inner_error_joint_plan import solve_interior_plan as joint_solve
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as prior
from scripts.tracking.trajectory_search import audit_velocity_warning_full_horizon as warning

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/full_horizon_feasibility_restoration_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/full_horizon_feasibility_restoration_20261110.md"
QA = REPO / "outputs/qa/full_horizon_feasibility_restoration_20261110/result.json"
DRY = REPO / "outputs/full_horizon_feasibility_restoration_dry_20261110/result.json"
OUTPUT = REPO / "outputs/full_horizon_feasibility_restoration_20261110"
WARNING_RESULT = {
    "path": str(REPO / "outputs/velocity_warning_full_horizon_20261110/result.json"),
    "sha256": "d30d1fc5ef37ec819c414c859554ffd377e9725917d77a6f98bb80d690e4f03c",
}
CASES = (2, 6)


def source_refs():
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        REPO / "src/s2a/research/trajectory_search/past_inner_error_joint_plan.py",
        REPO / "tests/research/native_runtime/test_full_horizon_restoration_inputs.py",
    )
    return warning.source_refs() + [prefix.reference(path) for path in paths]


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    c = prefix.read(config_ref)
    prefix.require(
        c
        == {
            "schema": "s2a.full-horizon-feasibility-restoration.v1",
            "cases": [2, 6],
            "sampling_seed": 43,
            "warning_steps": [39, 47],
            "chunk_indices": [4, 5],
            "cursor": 7,
            "branches": ["joint_only", "restore_seed", "restore_joint_warm"],
            "horizon": 8,
            "control_dt_s": 0.05,
            "inner_dt_s": 0.002,
            "joint_buffer_rad": 0.05,
            "braking_time_s": 0.1,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "max_models": 2,
            "max_forward_queries": 2,
            "max_optimizer_calls": 6,
            "max_optimizer_responses": 1600,
            "max_independent_endpoint_queries": 1200,
            "max_independent_transition_steps": 15600,
            "worker_timeout_seconds": 1800,
            "physics_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered feasibility-restoration diagnostic",
    )
    observed = prefix.read(WARNING_RESULT)
    prefix.require(
        observed["complete"] is observed["passed"] is observed["sources_unchanged"] is True,
        "Closed earlier warning model analysis",
    )
    inputs = warning.load_inputs()
    refs = inputs["source_sha256"]
    refs.update({str(CONFIG): config_ref["sha256"], WARNING_RESULT["path"]: WARNING_RESULT["sha256"]})
    cases = {}
    for ref in observed["case_refs"]:
        result = prefix.read(ref)
        refs[ref["path"]] = ref["sha256"]
        prefix.require(
            result["complete"] is result["passed"] is True and result["case_index"] in CASES,
            "Closed earlier warning case",
        )
        cases[result["case_index"]] = result["analysis"]
    prefix.require(set(cases) == set(CASES), "Exactly two previously analyzed warning states")
    for case in inputs["cases"]:
        previous = cases[case["spec"]["case_index"]]
        prefix.require(
            previous["warning"] == case["first_warning"]
            and previous["reference_chunk"] == case["intervention_spec"]
            and all(value <= 1e-8 for value in previous["reconstruction_max_difference"].values()),
            "Same causal state and generated reference",
        )
    return inputs


def with_all_tcp_errors(witness, targets, site_from_tcp):
    """Add model errors without treating an unconstrained plan as a quality witness."""
    prefix.require(witness.get("witness_found") is True, "A claimed joint witness is required")
    poses = np.asarray(witness["rollout"]["pose"])
    targets = np.asarray(targets)
    prefix.require(poses.shape == targets.shape == (8, 4, 4), "Eight aligned model poses")
    result = dict(witness)
    result["tcp_errors"] = np.asarray(
        [pose_error(target, pose @ site_from_tcp) for target, pose in zip(targets, poses, strict=True)]
    )
    return result


def analyze(case, stats):
    current, site_from_tcp, targets, args, differences = warning.restore_warning_context(case, stats)
    profile = np.zeros((25, 7))
    output, terminal_normals = {}, None

    def verify(candidate):
        stats["independent_endpoint_queries_charged"] += 200
        stats["independent_transition_steps_charged"] += 2600
        verified = audit_full_horizon_witness(
            candidate,
            current["mass"],
            current["jacobian"],
            current["contract"]["kp"],
            current["contract"]["kd"],
            current["q"],
            current["velocity"],
            case["initial_joint"],
            current["interface"],
            current["previous"][:6],
            current["site"],
            site_from_tcp,
            targets,
            current["limits"][:, 0],
            current["limits"][:, 1],
            past_inner_error_rad=profile,
            enforce_pose_caps=False,
        )
        prefix.require(
            verified["passed"] is True
            and verified["independent_endpoint_queries"] == 200
            and verified["independent_transition_steps"] == 2600,
            "Bound independent model audit",
        )
        errors = verified["tcp_errors"]
        slack = max(
            0.0,
            float(
                np.max(
                    np.r_[
                        np.linalg.norm(errors[:, :3], axis=1) / 0.01 - 1,
                        np.linalg.norm(errors[:, 3:], axis=1) / 0.05 - 1,
                    ]
                )
            ),
        )
        if "quality_slack" in candidate:
            np.testing.assert_allclose(slack, candidate["quality_slack"], atol=1e-8, rtol=0)
        verified["actual_quality_slack"] = slack
        return verified

    for label in ("joint_only", "restore_seed", "restore_joint_warm"):
        joint = output.get("joint_only", {})
        if label == "restore_joint_warm" and not joint.get("witness_found", False):
            output[label] = {"skipped": True, "reason": "joint_only_no_verified_witness"}
            continue
        extra = {"initial_guess": joint["planned_commands"]} if label == "restore_joint_warm" else {}
        solver = joint_solve if label == "joint_only" else solve_full_horizon
        if label != "joint_only":
            extra["feasibility_restoration"] = True
        stats["optimizer_calls_attempted"] += 1
        started = time.monotonic()
        solved = solver(*args, past_inner_error_rad=profile, **extra)
        stats["optimizer_calls_completed"] += 1
        stats["optimizer_responses"] += int(solved["response_evaluations"])
        prefix.require(
            solved["response_evaluations"] <= (160 if label == "joint_only" else 320), "Per-search response budget"
        )
        if terminal_normals is None:
            terminal_normals = solved["terminal_normals"]
        else:
            np.testing.assert_array_equal(terminal_normals, solved["terminal_normals"])
        candidate = None
        if label == "joint_only" and solved["witness_found"]:
            candidate = with_all_tcp_errors(solved, targets, site_from_tcp)
        elif label != "joint_only":
            candidate = solved.get("best_joint_safe")
        verification = None if candidate is None else verify(candidate)
        if label != "joint_only" and solved["witness_found"]:
            prefix.require(
                verification is not None and verification["all_pose_caps_satisfied"] is True,
                "Every accepted quality witness must independently satisfy all caps",
            )
        summary = {k: v for k, v in solved.items() if k not in ("rollout", "best_joint_safe")}
        if candidate is not None:
            summary["best_joint_safe"] = {k: v for k, v in candidate.items() if k != "rollout"}
            summary["quality_witness_verified"] = bool(verification["all_pose_caps_satisfied"])
        else:
            summary["quality_witness_verified"] = False
        summary.update(independent_verification=verification, elapsed_seconds=time.monotonic() - started)
        output[label] = summary
    return {
        "case_index": case["spec"]["case_index"],
        "warning": case["first_warning"],
        "reference_chunk": case["intervention_spec"],
        "reconstruction_max_difference": differences,
        "historical_initial_joint_available": False,
        "initial_past_error_available": False,
        "candidates": output,
        "scope": "static frozen model; no dispatch or physical qualification",
    }


def worker(index):
    def interrupted(signum, _frame):
        raise InterruptedError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        inputs["source_sha256"] == registration["source_sha256"]
        and inputs["asset_refs"] == registration["asset_refs"]
        and source_refs() == registration["source_refs"],
        "Frozen input and source identities",
    )
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
    stats = {
        "model_initializations_attempted": 0,
        "model_initializations_completed": 0,
        "forward_queries_attempted": 0,
        "forward_queries_completed": 0,
        "optimizer_calls_attempted": 0,
        "optimizer_calls_completed": 0,
        "optimizer_responses": 0,
        "baseline_rollouts": 0,
        "independent_endpoint_queries_charged": 0,
        "independent_transition_steps_charged": 0,
        "max_qpos_restore_difference": 0.0,
        "physics_steps": 0,
    }
    result = {
        "complete": False,
        "passed": False,
        "case_index": index,
        "stats": stats,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        result["osmesa_ref"] = prior.preload_software_gl(case, inputs["source_sha256"])
        result["analysis"] = analyze(case, stats)
        prefix.require(
            stats["model_initializations_completed"] == 1
            and stats["forward_queries_completed"] == 1
            and stats["baseline_rollouts"] == 1
            and 2 <= stats["optimizer_calls_completed"] <= 3
            and stats["optimizer_responses"] <= 800
            and stats["independent_endpoint_queries_charged"] <= 600
            and stats["independent_transition_steps_charged"] <= 7800,
            "Per-case budget",
        )
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / f"case_{index:02d}" / "result.json", result), flush=True)


def execute():
    prefix.require(not OUTPUT.exists(), "New output only; no automatic retry")
    dry_ref, qa_ref = prefix.reference(DRY), prefix.reference(QA)
    dry, qa = prefix.read(dry_ref), prefix.read(qa_ref)
    prefix.require(
        dry["complete"] is dry["passed"] is True and dry["models"] == dry["physics_steps"] == 0, "Zero-model dry closed"
    )
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "Full QA source identity")
    sources, inputs = source_refs(), load_inputs()
    prefix.require(
        dry["source_refs"] == sources
        and dry["source_sha256"] == inputs["source_sha256"]
        and dry["asset_refs"] == inputs["asset_refs"],
        "Dry source/input identity",
    )
    registration = publish(
        OUTPUT / "registration.json",
        {
            "dry_ref": dry_ref,
            "qa_ref": qa_ref,
            "warning_result_ref": WARNING_RESULT,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "cases": list(CASES),
            "max_models": 2,
            "max_forward_queries": 2,
            "max_optimizer_calls": 6,
            "max_optimizer_responses": 1600,
            "max_independent_endpoint_queries": 1200,
            "max_independent_transition_steps": 15600,
            "physics_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
    )
    specs = [
        {
            "key": f"case_{index:02d}",
            "argv": [
                sys.executable,
                "-m",
                "scripts.tracking.trajectory_search.audit_full_horizon_restoration",
                "--worker",
                str(index),
            ],
            "log": OUTPUT / f"case_{index:02d}" / "worker.log",
        }
        for index in CASES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=1800)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers_ref = publish(OUTPUT / "workers.json", lifecycle)
    result = {
        "complete": False,
        "passed": False,
        "registration_ref": registration,
        "workers_ref": workers_ref,
        "physics_steps": 0,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 2
            and all(
                w["returncode"] == 0 and w["actual_wait"] and w["reaped"] and not w["signal_sent"]
                for w in lifecycle["workers"]
            ),
            "Both workers closed and reaped",
        )
        refs = [prefix.reference(OUTPUT / f"case_{index:02d}" / "result.json") for index in CASES]
        rows = [prefix.read(ref) for ref in refs]
        prefix.require(all(row["complete"] is row["passed"] is True for row in rows), "Complete restoration audits")
        prefix.require(
            all(prefix.reference(path)["sha256"] == digest for path, digest in inputs["source_sha256"].items())
            and source_refs() == sources
            and snapshot(REPO)["sources"] == before,
            "Sources and inputs unchanged",
        )
        totals = {
            name: sum(row["stats"][name] for row in rows)
            for name in (
                "model_initializations_completed",
                "forward_queries_completed",
                "baseline_rollouts",
                "optimizer_calls_completed",
                "optimizer_responses",
                "independent_endpoint_queries_charged",
                "independent_transition_steps_charged",
            )
        }
        prefix.require(
            totals["model_initializations_completed"] == totals["forward_queries_completed"] == 2
            and totals["baseline_rollouts"] == 2
            and totals["optimizer_calls_completed"] <= 6
            and totals["optimizer_responses"] <= 1600
            and totals["independent_endpoint_queries_charged"] <= 1200
            and totals["independent_transition_steps_charged"] <= 15600,
            "Registered aggregate budget",
        )
        result.update(complete=True, passed=True, sources_unchanged=True, case_refs=refs, **totals)
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", type=int, choices=CASES)
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        prefix.require(not DRY.exists(), "New dry output only")
        inputs = load_inputs()
        print(
            publish(
                DRY,
                {
                    "complete": True,
                    "passed": True,
                    "source_refs": source_refs(),
                    "source_sha256": inputs["source_sha256"],
                    "asset_refs": inputs["asset_refs"],
                    "cases": list(CASES),
                    "models": 0,
                    "physics_steps": 0,
                    "world_model_calls": 0,
                    "fresh_cases": 0,
                },
            ),
            flush=True,
        )
    elif args.execute:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
