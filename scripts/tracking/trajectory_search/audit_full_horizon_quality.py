"""Compare all-point constrained plans at two actual first-warning states."""

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
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.trajectory_search.full_horizon_quality import solve_full_horizon
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as prior

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/full_horizon_quality_feasibility_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/full_horizon_quality_feasibility_20261110.md"
QA = REPO / "outputs/qa/full_horizon_quality_feasibility_20261110/result.json"
DRY = REPO / "outputs/full_horizon_quality_feasibility_dry_20261110/result.json"
OUTPUT = REPO / "outputs/full_horizon_quality_feasibility_20261110"
FIRST_POINT = {
    "path": str(REPO / "outputs/early_quality_feasibility_v2_20261110/result.json"),
    "sha256": "a035a84aaf1b98de08a6dab3d500ed177ec280b8f3f82cd5dd93178959760821",
}
CASES = (2, 6)


def source_refs():
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        REPO / "src/s2a/research/trajectory_search/full_horizon_quality.py",
        REPO / "src/s2a/research/joint_control/tracking/full_horizon_quality_audit.py",
        REPO / "tests/research/trajectory_search/test_full_horizon_quality.py",
        REPO / "tests/research/native_runtime/test_full_horizon_quality_inputs.py",
    )
    return prior.source_refs() + [prefix.reference(path) for path in paths]


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    c = prefix.read(config_ref)
    prefix.require(
        c
        == {
            "schema": "s2a.full-horizon-quality-feasibility.v1",
            "cases": [2, 6],
            "sampling_seed": 43,
            "start_steps": [43, 52],
            "branches": ["original_seed", "first_point_warm"],
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
            "max_optimizer_calls": 4,
            "max_optimizer_responses": 1280,
            "max_independent_endpoint_queries": 800,
            "max_independent_transition_steps": 10400,
            "worker_timeout_seconds": 1800,
            "physics_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered full-horizon diagnostic",
    )
    recorded = prefix.read(FIRST_POINT)
    prefix.require(
        recorded["complete"] is recorded["passed"] is recorded["sources_unchanged"] is True,
        "Closed original first-point diagnostic",
    )
    inputs = prior.load_inputs()
    refs = inputs["source_sha256"]
    refs.update({str(CONFIG): config_ref["sha256"], FIRST_POINT["path"]: FIRST_POINT["sha256"]})
    saved = {}
    for ref in recorded["case_refs"]:
        data = prefix.read(ref)
        refs[ref["path"]] = ref["sha256"]
        if data["offset"] == 0:
            prefix.require(
                data["case_index"] in CASES and data["complete"] is data["passed"] is True,
                "Closed current first-point model candidate",
            )
            saved[data["case_index"]] = data
    prefix.require(set(saved) == set(CASES), "One current first-point model candidate per case")
    for case in inputs["cases"]:
        index = case["spec"]["case_index"]
        candidate = saved[index]["analysis"]["candidates"]["nominal"]
        prefix.require(
            candidate["witness_found"] is True and candidate["independent_verification"]["passed"] is True,
            "Original first-point candidate independently verified",
        )
        np.testing.assert_array_equal(
            candidate["seed_commands"], case["feedback_periods"][0]["decision"]["refined_plan"]
        )
        case["first_point_warm"] = np.asarray(candidate["planned_commands"], dtype=float)
    return inputs


def analyze(case, stats):
    import mujoco

    def prohibited(*_args, **_kwargs):
        raise RuntimeError("Static full-horizon diagnostic forbids physical integration")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, prohibited)
    stats["model_initializations_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    stats["model_initializations_completed"] += 1
    data = mujoco.MjData(model)
    current = prior.restored_context(model, data, case, case["feedback_periods"], 0, stats)
    site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    cursor = case["intervention_spec"]["cursor"]
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
    error = np.zeros((25, 7))
    output, normals = {}, None
    for label in ("original_seed", "first_point_warm"):
        extra = {"initial_guess": case["first_point_warm"]} if label == "first_point_warm" else {}
        stats["optimizer_calls_attempted"] += 1
        started = time.monotonic()
        solved = solve_full_horizon(*args, past_inner_error_rad=error, **extra)
        stats["optimizer_calls_completed"] += 1
        stats["optimizer_responses"] += int(solved["response_evaluations"])
        prefix.require(solved["response_evaluations"] <= 320, "Per-search response budget")
        if normals is None:
            normals = solved["terminal_normals"]
        else:
            np.testing.assert_array_equal(normals, solved["terminal_normals"])
        verification = None
        if solved["witness_found"]:
            stats["independent_endpoint_queries_charged"] += 200
            stats["independent_transition_steps_charged"] += 2600
            verification = audit_full_horizon_witness(
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
                targets,
                current["limits"][:, 0],
                current["limits"][:, 1],
                past_inner_error_rad=error,
            )
            prefix.require(
                verification["independent_endpoint_queries"] == 200
                and verification["independent_transition_steps"] == 2600
                and verification["all_pose_caps_satisfied"] is True,
                "Bound full-horizon witness audit",
            )
        output[label] = {k: v for k, v in solved.items() if k != "rollout"}
        output[label].update(independent_verification=verification, elapsed_seconds=time.monotonic() - started)
    return {
        "case_index": case["spec"]["case_index"],
        "start_step": case["spec"]["prefix_steps"],
        "initial_past_error_available": False,
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
            and stats["optimizer_calls_completed"] == 2
            and stats["optimizer_responses"] <= 640
            and stats["independent_endpoint_queries_charged"] <= 400
            and stats["independent_transition_steps_charged"] <= 5200,
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
            "first_point_ref": FIRST_POINT,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "cases": list(CASES),
            "max_models": 2,
            "max_forward_queries": 2,
            "max_optimizer_calls": 4,
            "max_optimizer_responses": 1280,
            "max_independent_endpoint_queries": 800,
            "max_independent_transition_steps": 10400,
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
                "scripts.tracking.trajectory_search.audit_full_horizon_quality",
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
            totals["model_initializations_completed"] == totals["forward_queries_completed"] == 2
            and totals["baseline_rollouts"] == 2
            and totals["optimizer_calls_completed"] == 4
            and totals["optimizer_responses"] <= 1280
            and totals["independent_endpoint_queries_charged"] <= 800
            and totals["independent_transition_steps_charged"] <= 10400,
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
                    "start_steps": [case["spec"]["prefix_steps"] for case in inputs["cases"]],
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
