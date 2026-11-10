"""Audit model feasibility at two actual quality-feedback stop states."""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

from s2a.geometry import inverse
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.generated_twist import reference_world_twist
from s2a.research.joint_control.tracking.early_quality_diagnostic import (
    completed_error_profile,
    independently_check_witness,
)
from s2a.research.joint_control.tracking.quality_stop_diagnostic import stopped_feedback_inputs
from s2a.research.trajectory_search.past_inner_error_joint_plan import solve_interior_plan as joint_solve
from s2a.research.trajectory_search.quality_capped_inner_joint_plan import solve_interior_plan as quality_solve
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import compare_early_warning_tracking as early
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as prior

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/quality_stop_feasibility_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/quality_stop_feasibility_20261110.md"
QA = REPO / "outputs/qa/quality_stop_feasibility_20261110/result.json"
DRY = REPO / "outputs/quality_stop_feasibility_dry_20261110/result.json"
OUTPUT = REPO / "outputs/quality_stop_feasibility_20261110"
PRODUCER = {
    "path": str(REPO / "outputs/early_quality_feedback_20261110/producer/result.json"),
    "sha256": "0fb0e65e4dab9cfa2ce3e94001b2edea8ebf29fb8a5b23e473213e1c0e8a87c1",
}
CHECKER = {
    "path": str(REPO / "outputs/early_quality_feedback_20261110/checker/result.json"),
    "sha256": "2c64fd0e409a27318b719a0cc35b72eca52b31a4c209977cf14b3d482b1e5cb7",
}
CASES = (2, 6)


def source_refs():
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        REPO / "src/s2a/research/joint_control/tracking/quality_stop_diagnostic.py",
        REPO / "src/s2a/research/trajectory_search/past_inner_error_joint_plan.py",
        REPO / "tests/research/joint_control/tracking/test_quality_stop_diagnostic.py",
        REPO / "tests/research/native_runtime/test_quality_stop_feasibility_inputs.py",
    )
    return prior.source_refs() + [prefix.reference(path) for path in paths]


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    c = prefix.read(config_ref)
    prefix.require(
        c
        == {
            "schema": "s2a.quality-stop-feasibility.v1",
            "cases": [2, 6],
            "completed_periods": 5,
            "sampling_seed": 43,
            "branches": ["quality_past", "quality_nominal", "joint_past", "quality_past_warm"],
            "control_dt_s": 0.05,
            "inner_dt_s": 0.002,
            "joint_buffer_rad": 0.05,
            "braking_time_s": 0.1,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "max_models": 2,
            "max_forward_queries": 12,
            "max_optimizer_calls": 8,
            "max_optimizer_responses": 1280,
            "max_independent_endpoint_queries": 1600,
            "max_independent_transition_steps": 20800,
            "worker_timeout_seconds": 1800,
            "physics_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered stop-state diagnostic",
    )
    producer, checker = prefix.read(PRODUCER), prefix.read(CHECKER)
    prefix.require(
        producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
        and checker["complete"] is checker["passed"] is checker["sources_unchanged"] is True,
        "Both quality feedback phases must be closed",
    )
    inputs = early.load_inputs()
    refs = inputs["source_sha256"]
    refs.update(
        {
            str(CONFIG): config_ref["sha256"],
            PRODUCER["path"]: PRODUCER["sha256"],
            CHECKER["path"]: CHECKER["sha256"],
            prior.OSMESA["path"]: prior.OSMESA["sha256"],
        }
    )
    prefix.checked(prior.OSMESA)
    branches = {}
    for owner in (producer, checker):
        for ref in owner["branch_refs"]:
            data = prefix.read(ref)
            refs[ref["path"]] = ref["sha256"]
            key = (data["phase"], data["case_index"], data["branch"])
            prefix.require(
                key not in branches and data["complete"] is data["passed"] is True, "Unique closed feedback branch"
            )
            branches[key] = data
    for case in inputs["cases"]:
        index = case["spec"]["case_index"]
        produced, checked = branches[("producer", index, "quality")], branches[("checker", index, "quality")]
        p, q = produced["runtime"]["continuation"], checked["runtime"]["continuation"]
        prefix.require(
            p["stop"]["reason"] == "quality_no_witness"
            and p["stop"] == q["stop"]
            and q["decision_stop_recorded_not_recomputed"] is True
            and not checked["quality_module_imported"]
            and len(p["periods"]) == len(q["periods"]) == 5,
            "Independently replayed bounded stopped feedback",
        )
        for left, right in zip(p["periods"], q["periods"], strict=True):
            np.testing.assert_array_equal(left["command"], right["command"])
            np.testing.assert_array_equal(left["actual_state"], right["actual_state"])
        case["feedback_periods"] = p["periods"]
        case["stop_inputs"] = stopped_feedback_inputs(p["periods"], case["spec"]["prefix_steps"])
        case["initial_joint"] = np.asarray(produced["runtime"]["runtime_initial_joint"], dtype=float)
        prefix.require(case["config"]["osmesa_library"] == prior.OSMESA["path"], "Bound software GL")
    return inputs


def analyze(case, stats):
    import mujoco

    def prohibited(*_args, **_kwargs):
        raise RuntimeError("Static stop-state diagnostic forbids physical integration")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, prohibited)
    stats["model_initializations_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    stats["model_initializations_completed"] += 1
    data = mujoco.MjData(model)
    periods, stopped = case["feedback_periods"], case["stop_inputs"]
    synthetic = periods + [{"step": stopped["stop_step"], "decision": {"refined_plan": stopped["seed"]}}]
    cache = {}

    def context(index):
        if index not in cache:
            cache[index] = prior.restored_context(model, data, case, synthetic, index, stats)
        return cache[index]

    profile = completed_error_profile(
        periods, case["spec"]["prefix_steps"], 5, lambda index: context(index)["rollout"]["inner_joint_position"][:25]
    )
    np.testing.assert_allclose(profile["inner_position_error_rad"], stopped["past_inner_error_rad"], atol=1e-10, rtol=0)
    current = context(5)
    np.testing.assert_array_equal(current["state"], stopped["state"])
    np.testing.assert_array_equal(current["previous"], stopped["previous"])
    site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    cursor = case["intervention_spec"]["cursor"] + 5
    targets = case["targets"][cursor : cursor + 8]
    twists = np.asarray(
        [
            reference_world_twist(case["targets"], index, case["anchor"], 0.05)["twist"]
            for index in range(cursor, cursor + 8)
        ]
    )
    args = (
        current["model"],
        current["velocity"],
        current["plan"],
        current["previous"][:6],
        current["interface"],
        current["site"],
        site_from_tcp,
        targets,
        twists,
        current["limits"][:, 0],
        current["limits"][:, 1],
    )
    output, normals = {}, None
    for label in ("quality_past", "quality_nominal", "joint_past", "quality_past_warm"):
        warm = output.get("joint_past", {})
        if label == "quality_past_warm" and not warm.get("witness_found", False):
            output[label] = {"skipped": True, "reason": "joint_past_no_verified_witness"}
            continue
        error = np.zeros((25, 7)) if label == "quality_nominal" else profile["inner_position_error_rad"]
        solver = joint_solve if label == "joint_past" else quality_solve
        extra = {"initial_guess": warm["planned_commands"]} if label == "quality_past_warm" else {}
        stats["optimizer_calls_attempted"] += 1
        started = time.monotonic()
        solved = solver(*args, past_inner_error_rad=error, **extra)
        stats["optimizer_calls_completed"] += 1
        stats["optimizer_responses"] += int(solved["response_evaluations"])
        prefix.require(solved["response_evaluations"] <= 160, "Original per-search response budget")
        if normals is None:
            normals = solved["terminal_normals"]
        else:
            np.testing.assert_array_equal(normals, solved["terminal_normals"])
        verification = None
        if solved["witness_found"]:
            stats["independent_endpoint_queries_charged"] += 200
            stats["independent_transition_steps_charged"] += 2600
            verification = independently_check_witness(
                solved,
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
                targets[0],
                current["limits"][:, 0],
                current["limits"][:, 1],
                past_inner_error_rad=error,
                enforce_pose_caps=label != "joint_past",
            )
            prefix.require(
                verification["independent_endpoint_queries"] == 200
                and verification["independent_transition_steps"] == 2600,
                "Bound witness audit cost",
            )
        output[label] = {k: v for k, v in solved.items() if k != "rollout"}
        output[label].update(independent_verification=verification, elapsed_seconds=time.monotonic() - started)
    return {
        "case_index": case["spec"]["case_index"],
        "stop_step": stopped["stop_step"],
        "completed_periods": profile["completed_periods"],
        "past_error_profile": profile["inner_position_error_rad"],
        "shifted_seed": stopped["seed"],
        "previous_command": stopped["previous"],
        "candidates": output,
        "scope": "static frozen model only; no dispatch or physical qualification",
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
    directory = OUTPUT / f"case_{index:02d}"
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
            and stats["forward_queries_completed"] == 6
            and stats["baseline_rollouts"] == 6
            and 3 <= stats["optimizer_calls_completed"] <= 4
            and stats["optimizer_responses"] <= 640
            and stats["independent_endpoint_queries_charged"] <= 800
            and stats["independent_transition_steps_charged"] <= 10400,
            "Per-case budget",
        )
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


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
            "producer_ref": PRODUCER,
            "checker_ref": CHECKER,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "cases": list(CASES),
            "max_models": 2,
            "max_forward_queries": 12,
            "max_optimizer_calls": 8,
            "max_optimizer_responses": 1280,
            "max_independent_endpoint_queries": 1600,
            "max_independent_transition_steps": 20800,
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
                "scripts.tracking.trajectory_search.audit_quality_stop_feasibility",
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
        prefix.require(all(row["complete"] is row["passed"] is True for row in rows), "Complete state audits")
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
            totals["model_initializations_completed"] == 2
            and totals["forward_queries_completed"] == 12
            and totals["baseline_rollouts"] == 12
            and totals["optimizer_calls_completed"] <= 8
            and totals["optimizer_responses"] <= 1280
            and totals["independent_endpoint_queries_charged"] <= 1600
            and totals["independent_transition_steps_charged"] <= 20800,
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
                    "stop_steps": [case["stop_inputs"]["stop_step"] for case in inputs["cases"]],
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
