"""Search full-horizon quality at two earlier causal velocity warnings."""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

from s2a.geometry import inverse, pose_error
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.generated_twist import reference_world_twist
from s2a.research.joint_control.tracking.early_quality_diagnostic import independently_check_witness
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.joint_control.tracking.velocity_warning_inputs import (
    current_recorded_decision,
    first_velocity_warning,
)
from s2a.research.trajectory_search.full_horizon_quality import solve_full_horizon
from s2a.research.trajectory_search.quality_capped_inner_joint_plan import solve_interior_plan
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import compare_early_warning_tracking as early
from scripts.simulation.runtime import compare_multicase_emergency_retreat as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as prior
from scripts.tracking.trajectory_search import audit_full_horizon_quality as full

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/velocity_warning_full_horizon_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/velocity_warning_full_horizon_20261110.md"
QA = REPO / "outputs/qa/velocity_warning_full_horizon_20261110/result.json"
DRY = REPO / "outputs/velocity_warning_full_horizon_dry_20261110/result.json"
OUTPUT = REPO / "outputs/velocity_warning_full_horizon_20261110"
CASES = ((2, 39, 4), (6, 47, 5))


def source_refs():
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        REPO / "src/s2a/research/joint_control/tracking/velocity_warning_inputs.py",
        REPO / "tests/research/joint_control/tracking/test_velocity_warning_inputs.py",
        REPO / "tests/research/native_runtime/test_velocity_warning_full_horizon_inputs.py",
    )
    return full.source_refs() + [prefix.reference(path) for path in paths]


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    c = prefix.read(config_ref)
    prefix.require(
        c
        == {
            "schema": "s2a.velocity-warning-full-horizon.v1",
            "cases": [2, 6],
            "sampling_seed": 43,
            "warning_steps": [39, 47],
            "chunk_indices": [4, 5],
            "cursor": 7,
            "branches": ["full_original", "first_quality", "full_first_warm"],
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
        "Exact registered velocity-warning diagnostic",
    )
    warning, produced, checked = (prefix.read(ref) for ref in (early.AUDIT, prior.PRODUCER, prior.CHECKER))
    prefix.require(
        warning["complete"] is warning["passed"] is warning["sources_unchanged"] is True
        and produced["complete"] is produced["passed"] is produced["sources_unchanged"] is True
        and checked["complete"] is checked["passed"] is checked["sources_unchanged"] is True,
        "Warning audit and reference replays closed",
    )
    inputs = prefix.load_inputs({2: 39, 6: 47})
    refs = inputs["source_sha256"]
    for ref in (config_ref, early.AUDIT, prior.PRODUCER, prior.CHECKER, prior.OSMESA):
        prefix.checked(ref)
        refs[ref["path"]] = ref["sha256"]
    review = prefix.read(prefix.REVIEW)
    source_map = review["source_and_artifact_sha256"]
    previous_runs = {}
    for owner in (produced, checked):
        for ref in owner["branch_refs"]:
            data = prefix.read(ref)
            refs[ref["path"]] = ref["sha256"]
            if data["branch"] == "refiner" and data["case_index"] in (2, 6):
                previous_runs[(data["phase"], data["case_index"])] = data
    prefix.require(len(previous_runs) == 4, "Independent replayed controller initial joints")
    for case in inputs["cases"]:
        index, n = case["spec"]["case_index"], case["spec"]["prefix_steps"]
        item = next(item for item in CASES if item[0] == index)
        prefix.require(n == item[1], "Exact first warning endpoint")
        warning_ref = warning["case_refs"][index]
        warning_case = prefix.read(warning_ref)
        refs[warning_ref["path"]] = warning_ref["sha256"]
        rule = warning_case["rules"]["velocity_0p2s"]
        causal = first_velocity_warning(case["saved"]["sim_states"], case["record_xml"])
        prefix.require(
            rule["first_alarm_endpoint"] == rule["first_alarm_while_buffer_safe_before_first_outside"] == n
            and causal["endpoint"] == n,
            "Existing first causal warning",
        )
        folder = Path(review["cases"][index]["mechanical"]["case"])
        prefix.require(folder.is_relative_to(REPO / "outputs"), "Reviewed local development directory")
        chunk_path, tool_path = folder / "chunks.json", Path(case["config"]["tool_contract"])
        for path in (chunk_path, tool_path):
            ref = prefix.reference(path)
            prefix.require(source_map.get(ref["path"]) == ref["sha256"], "Reviewed reference/tool identity")
            refs[ref["path"]] = ref["sha256"]
        row = prefix.read_recorded_prefix(folder / "steps.jsonl", n + 1, 600)[-1]
        decision = current_recorded_decision(row, step=n, previous_command=case["commands"][-1])
        chunk = previous.read_array_item(chunk_path, item[2])
        prefix.require(
            chunk["step"] == item[2] * 8
            and decision["chunk_index"] == item[2]
            and decision["cursor"] == 7
            and chunk["step"] + 7 == n,
            "Only the reference chunk already generated at the warning",
        )
        targets, anchor = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["anchor_tcp"])
        tool = np.asarray(json.loads(tool_path.read_text())["source_from_tcp"])
        prefix.require(
            targets.shape == (40, 4, 4)
            and anchor.shape == tool.shape == (4, 4)
            and np.isfinite(targets).all()
            and np.isfinite(anchor).all()
            and np.isfinite(tool).all(),
            "One finite already-generated reference",
        )
        p, q = previous_runs[("producer", index)], previous_runs[("checker", index)]
        initial = np.asarray(p["runtime"]["runtime_initial_joint"])
        np.testing.assert_array_equal(initial, q["runtime"]["runtime_initial_joint"])
        prefix.require(
            p["runtime"]["initial_joint_historical_value_available"] is False,
            "Original initial-joint logging limitation preserved",
        )
        case.update(
            initial_joint=initial,
            targets=targets,
            anchor=anchor,
            tool=tool,
            decision=decision,
            intervention_spec={"cursor": 7, "chunk_index": item[2]},
            first_warning=causal,
            feedback_periods=[{"step": n, "decision": {"refined_plan": decision["plan"]}}],
        )
    return inputs


def restore_warning_context(case, stats):
    """Restore only the current robot and reference, without stepping physics."""
    import mujoco

    def prohibited(*_args, **_kwargs):
        raise RuntimeError("Static warning diagnostic forbids physical integration")

    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, prohibited)
    stats["model_initializations_attempted"] += 1
    model = mujoco.MjModel.from_xml_string(case["record_xml"])
    stats["model_initializations_completed"] += 1
    current = prior.restored_context(model, mujoco.MjData(model), case, case["feedback_periods"], 0, stats)
    logged = case["decision"]
    nominal = np.r_[pose_error(current["rollout"]["pose"][0], current["site"]), current["rollout"]["twist"][0]]
    differences = {
        key: float(np.max(np.abs(current[key] - logged[key]))) for key in ("mass", "jacobian", "q", "velocity", "site")
    }
    differences["twist"] = float(np.max(np.abs(current["jacobian"] @ current["velocity"] - logged["twist"])))
    differences["nominal_response"] = float(np.max(np.abs(nominal - logged["nominal_response"])))
    prefix.require(all(value <= 1e-8 for value in differences.values()), "Recorded current dynamics and prediction")
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
    return current, site_from_tcp, targets, args, differences


def analyze(case, stats):
    current, site_from_tcp, targets, args, differences = restore_warning_context(case, stats)
    profile = np.zeros((25, 7))
    output, terminal_normals = {}, None
    for label in ("full_original", "first_quality", "full_first_warm"):
        previous_first = output.get("first_quality", {})
        if label == "full_first_warm" and not previous_first.get("witness_found", False):
            output[label] = {"skipped": True, "reason": "first_quality_no_verified_witness"}
            continue
        solver = solve_interior_plan if label == "first_quality" else solve_full_horizon
        extra = {"initial_guess": previous_first["planned_commands"]} if label == "full_first_warm" else {}
        stats["optimizer_calls_attempted"] += 1
        started = time.monotonic()
        solved = solver(*args, past_inner_error_rad=profile, **extra)
        stats["optimizer_calls_completed"] += 1
        stats["optimizer_responses"] += int(solved["response_evaluations"])
        prefix.require(
            solved["response_evaluations"] <= (160 if label == "first_quality" else 320), "Per-search response budget"
        )
        if terminal_normals is None:
            terminal_normals = solved["terminal_normals"]
        else:
            np.testing.assert_array_equal(terminal_normals, solved["terminal_normals"])
        verification = None
        if solved["witness_found"]:
            stats["independent_endpoint_queries_charged"] += 200
            stats["independent_transition_steps_charged"] += 2600
            verifier = independently_check_witness if label == "first_quality" else audit_full_horizon_witness
            reference = targets[0] if label == "first_quality" else targets
            verification = verifier(
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
                reference,
                current["limits"][:, 0],
                current["limits"][:, 1],
                past_inner_error_rad=profile,
            )
            prefix.require(
                verification["independent_endpoint_queries"] == 200
                and verification["independent_transition_steps"] == 2600
                and verification["passed"] is True,
                "Bound independent model audit",
            )
        output[label] = {k: v for k, v in solved.items() if k != "rollout"}
        output[label].update(independent_verification=verification, elapsed_seconds=time.monotonic() - started)
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
            "warning_ref": early.AUDIT,
            "producer_ref": prior.PRODUCER,
            "checker_ref": prior.CHECKER,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "cases": [list(item) for item in CASES],
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
                "scripts.tracking.trajectory_search.audit_velocity_warning_full_horizon",
                "--worker",
                str(index),
            ],
            "log": OUTPUT / f"case_{index:02d}" / "worker.log",
        }
        for index, _n, _chunk in CASES
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
        refs = [prefix.reference(OUTPUT / f"case_{index:02d}" / "result.json") for index, _n, _chunk in CASES]
        rows = [prefix.read(ref) for ref in refs]
        prefix.require(all(row["complete"] is row["passed"] is True for row in rows), "Complete warning audits")
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
    group.add_argument("--worker", type=int, choices=[2, 6])
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
                    "cases": [list(item) for item in CASES],
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
