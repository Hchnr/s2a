"""Bind the fixed correct-reference and development shadow inputs."""

import json
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/bounded_release_revision_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/bounded_release_revision_20261111.md"
QA = REPO / "outputs/qa/bounded_release_revision_shadow_20261111/result.json"
DRY = REPO / "outputs/bounded_release_revision_shadow_dry_20261111/result.json"
ROOT = REPO / "outputs/bounded_release_revision_shadow_20261111"
ORACLE_ROOT = REPO / "outputs/release_floor_correct_reference_20261028"
DEVELOPMENT_ROOT = REPO / "outputs/tracking_preview_wam_development_v4_20261027/development/cases"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__),
            REPO / "src/s2a/research/wam/tracking/preview/bounded_release_revision.py",
            REPO / "scripts/wam/tracking/preview/replay_bounded_release_revision.py",
            REPO / "scripts/verification/reference_ports/check_bounded_release_revision.py",
            REPO / "tests/research/wam/tracking/preview/test_bounded_release_revision.py",
            REPO / "tests/research/wam/tracking/preview/test_bounded_release_revision_shadow.py",
        )
    ]


def configured_ref(value):
    path = REPO / value["path"]
    ref = {"path": str(path), "sha256": value["sha256"]}
    prefix.checked(ref)
    return ref


def expected_development_case(index):
    task, within = divmod(index, 4)
    return f"task{task}_init0_s{(42, 43)[within // 2]}_{('raw', 'terminal058')[within % 2]}"


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        config["schema"] == "s2a.bounded-release-revision.v1"
        and config["stride"] == config["bridge_steps"] == 8
        and config["control_dt_s"] == 0.05
        and config["release_clearance"] == config["opening_revision_gap"] == 0.04
        and config["current_position_gate_m"] == 0.03
        and config["current_rotation_gate_rad"] == 0.15
        and config["joint_buffer_rad"] == 0.05
        and config["max_holds_per_closing_episode"] == 1
        and config["correct_reference_cases"] == 18
        and config["correct_reference_steps"] == 4907
        and config["development_cases"] == 12
        and config["development_steps_per_case"] == 600
        and config["development_chunks_per_case"] == 75
        and config["shadow_max_workers"] == 32
        and config["world_model_calls"] == config["physics_steps"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "Exact registered release revision, matrix and budget",
    )
    fixed = {
        key: configured_ref(config[key])
        for key in (
            "correct_reference_qualification",
            "development_registration",
            "development_result",
            "studied_branch_producer",
            "studied_branch_checker",
        )
    }
    closed = prefix.read(fixed["correct_reference_qualification"])
    prefix.require(
        closed["complete"] is closed["passed"] is closed["performance_qualified"] is True
        and closed["qualified_cases"] == len(closed["cases"]) == 18
        and closed["candidate_physics_steps"] == closed["replay_physics_steps"] == 4907,
        "Full original release-floor correct-reference qualification",
    )
    prepared = prefix.read(closed["prepared_ref"])
    prefix.require(
        prepared["complete"] is prepared["passed"] is True and len(prepared["profiles"]) == 18, "Original profiles"
    )
    oracle = []
    for index, (item, profile) in enumerate(zip(closed["cases"], prepared["profiles"], strict=True)):
        prefix.require(item["case_index"] == profile["case_index"] == index, "Original correct-reference order")
        check = prefix.read(item["verified_ref"])
        robot = prefix.read(item["audit_refs"]["robot_inputs"])
        labels = prefix.read(item["audit_refs"]["labels_images"])
        result = prefix.read(check["result_ref"])
        prefix.require(
            check["complete"] is check["passed"] is check["performance_qualified"] is True
            and robot["complete"] is robot["passed"] is labels["complete"] is labels["passed"] is True
            and check["result_ref"] == robot["result_ref"] == labels["result_ref"]
            and result["complete"] is result["passed"] is result["performance_qualified"] is True
            and result["case_index"] == index
            and result["policy"] == "candidate"
            and result["physics_steps"] == item["physics_steps"] == profile["profile"]["physics_steps"]
            and robot["rows"] == item["physics_steps"],
            "Qualified exact candidate, robot input audit and profile",
        )
        directory = ORACLE_ROOT / f"case{index:02d}_candidate"
        rows_ref = robot["rows_ref"]
        prefix.require(Path(rows_ref["path"]) == directory / "steps.json", "Audited original rows")
        refs = {
            "result": check["result_ref"],
            "rows": rows_ref,
            "chunks": prefix.reference(directory / "chunks.json"),
            "tool": profile["tool_ref"],
            "trajectory": profile["profile"]["trajectory_ref"],
            "native_check": item["verified_ref"],
            "robot_audit": item["audit_refs"]["robot_inputs"],
            "label_audit": item["audit_refs"]["labels_images"],
        }
        for ref in refs.values():
            prefix.checked(ref)
        prefix.require(
            result["method_contract"]["tool_ref"] == refs["tool"]
            and Path(refs["trajectory"]["path"]).is_relative_to(REPO / "outputs"),
            "Original tool and repository-local correct reference",
        )
        oracle.append(
            {
                "family": "correct_reference",
                "index": index,
                "case_id": f"correct_{index:02d}",
                "steps": item["physics_steps"],
                "format": profile["profile"]["format"],
                "refs": refs,
            }
        )
    prefix.require(sum(item["steps"] for item in oracle) == 4907, "Unchanged complete correct-reference budget")

    registration, prior = prefix.read(fixed["development_registration"]), prefix.read(fixed["development_result"])
    prefix.require(
        prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True
        and prior["registration_ref"] == fixed["development_registration"]
        and len(registration["cases"]) == len(prior["case_refs"]) == 12,
        "Original complete reviewed development reference analysis",
    )
    development = []
    for index, (item, result_ref) in enumerate(zip(registration["cases"], prior["case_refs"], strict=True)):
        previous = prefix.read(result_ref)
        case, refs = item["case"], item["refs"]
        prefix.require(
            case["case_index"] == index
            and case["case_id"] == expected_development_case(index)
            and previous["complete"] is previous["passed"] is True
            and previous["data"]["case"] == case
            and previous["input_refs"] == refs
            and Path(refs["chunks"]["path"]).is_relative_to(DEVELOPMENT_ROOT / case["case_id"]),
            "Original development identity and reviewed input references",
        )
        for ref in refs.values():
            prefix.checked(ref)
        development.append(
            {"family": "development", "index": index, "case_id": case["case_id"], "steps": 600, "refs": refs}
        )
    return {
        "config_ref": config_ref,
        "qualification_ref": fixed["correct_reference_qualification"],
        "prepared_ref": closed["prepared_ref"],
        "development_registration_ref": fixed["development_registration"],
        "development_result_ref": fixed["development_result"],
        "studied_producer_ref": fixed["studied_branch_producer"],
        "studied_checker_ref": fixed["studied_branch_checker"],
        "cases": oracle + development,
    }


def read_case(item):
    """Validate the saved timeline, origin and all original executable endpoints."""
    refs, count = item["refs"], item["steps"]
    chunks, tool = prefix.read(refs["chunks"]), np.asarray(prefix.read(refs["tool"])["source_from_tcp"], dtype=float)
    if item["family"] == "correct_reference":
        rows = prefix.read(refs["rows"])
    else:
        with prefix.checked(refs["steps"]).open() as stream:
            rows = [json.loads(line) for line in stream]
    prefix.require(len(rows) == count and len(chunks) == (count + 7) // 8, "Complete baseline timeline")
    prefix.require(tool.shape == (4, 4) and np.isfinite(tool).all(), "Original TCP transform")
    source_poses = source_openings = None
    if item["family"] == "correct_reference":
        with np.load(prefix.checked(refs["trajectory"]), allow_pickle=False) as data:
            if item["format"] == "native_endpoints":
                source_poses, source_openings = data["source_pose"][1:].copy(), data["opening"][1:].copy()
            else:
                prefix.require(item["format"] == "world_reference", "Known original reference format")
                source_poses, source_openings = data["poses"].copy(), data["openings"].copy()
        prefix.require(
            len(source_poses) >= count and len(source_openings) >= count, "Complete original correct reference"
        )
    for query, chunk in enumerate(chunks):
        step, n = query * 8, len(chunk["openings"])
        poses, openings = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["openings"])
        prefix.require(
            chunk["step"] == step and poses.shape == (n, 4, 4) and n >= min(8, count - step),
            "Original generation clock and complete available prefix",
        )
        np.testing.assert_allclose(
            np.asarray(rows[step]["measured_source"]) @ tool, chunk["anchor_tcp"], atol=1e-10, rtol=0
        )
        if item["family"] == "correct_reference":
            np.testing.assert_allclose(poses, source_poses[step : step + n] @ tool, atol=1e-10, rtol=0)
            np.testing.assert_allclose(openings, source_openings[step : step + n], atol=1e-12, rtol=0)
        else:
            np.testing.assert_allclose(rows[step]["opening_before"], chunk["anchor_opening"], atol=1e-12, rtol=0)
        for offset in range(min(8, count - step)):
            row = rows[step + offset]
            prefix.require(row["step"] == step + offset and row["chunk_index"] == query, "Original action clock")
            if item["family"] == "correct_reference":
                prefix.require(row["reference_index"] == offset, "Original oracle reference cursor")
                executed_opening = row["gripper"]["preview"]["original_reference"]
                executed_pose = row["desired_tcp"]
            else:
                prefix.require(row["chunk_physical_step"] == offset, "Original development reference cursor")
                executed_opening = row["opening_target"]
                executed_pose = row["effective_desired_tcp"]
            np.testing.assert_allclose(executed_opening, openings[offset], atol=1e-12, rtol=0)
            np.testing.assert_allclose(executed_pose, poses[offset], atol=1e-10, rtol=0)
    return chunks, rows, tool


def read_studied(inputs):
    """Bind the previously replayed completed physical period before step368."""
    item = inputs["cases"][18 + 5]
    prefix.require(item["case_id"] == "task1_init0_s42_terminal058", "Registered studied development case")
    chunks, _, _ = read_case(item)
    produced, checked = (prefix.read(inputs[key]) for key in ("studied_producer_ref", "studied_checker_ref"))
    prefix.require(
        produced["complete"] is produced["passed"] is checked["complete"] is checked["passed"] is True
        and produced["branch"] == checked["branch"] == "remembered",
        "Closed independent remembered-reference physical branches",
    )
    runtime, replay = produced["runtime"], checked["runtime"]
    a, b = runtime["continuation"]["periods"][7], replay["continuation"]["periods"][7]
    prefix.require(
        a["step"] == b["step"] == 367 and len(a["actual_inner"]) == len(b["actual_inner"]) == 25,
        "Completed prior period",
    )
    for key in ("command", "actual_state", "actual_inner"):
        np.testing.assert_allclose(
            a[key] if key != "actual_inner" else [v["q"] for v in a[key]],
            b[key] if key != "actual_inner" else [v["q"] for v in b[key]],
            atol=1e-10,
            rtol=0,
        )
    for key in ("qvel", "time"):
        np.testing.assert_allclose(
            [v[key] for v in a["actual_inner"]], [v[key] for v in b["actual_inner"]], atol=1e-10, rtol=0
        )
    pa, pb = a["completed_feedback"]["scores"], b["completed_feedback"]["scores"]
    np.testing.assert_allclose(pa["actual_tcp"], pb["actual_tcp"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(pa["gripper"]["opening"], pb["gripper"]["opening"], atol=1e-12, rtol=0)
    limits = np.asarray(runtime["final_prefix_inner"]["limits"], dtype=float)
    joints = np.asarray([sample["q"] for sample in a["actual_inner"]], dtype=float)
    prefix.require(limits.shape == (7, 2) and joints.shape == (25, 7), "Registered actual joint samples and limits")
    margin = float(np.min(np.minimum(joints - limits[:, 0], limits[:, 1] - joints)))
    np.testing.assert_allclose(margin, a["minimum_inner_margin_rad"], atol=1e-10, rtol=0)
    old, latest = chunks[43], chunks[45]
    prefix.require(old["step"] == 344 and latest["step"] == 360, "Only already generated source chunks")
    np.testing.assert_allclose(pa["opening_targets"]["remembered"], old["openings"][23], atol=1e-12, rtol=0)
    np.testing.assert_allclose(pa["opening_targets"]["latest"], latest["openings"][7], atol=1e-12, rtol=0)
    latch = a["decision"]["gripper_decision"]["latch"]
    prefix.require(type(latch["closing_episode"]) is bool, "Completed gripper latch state")
    return {
        "step": 368,
        "old": old,
        "latest": latest,
        "measured_tcp": pa["actual_tcp"],
        "measured_opening": pa["gripper"]["opening"],
        "closing_episode": latch["closing_episode"],
        "previous_inner_margin_rad": margin,
    }


def dry_run():
    prefix.environment()
    inputs, sources = load_inputs(), source_refs()
    return publish(
        DRY,
        {
            "complete": True,
            "passed": True,
            "inputs": inputs,
            "source_refs": sources,
            "model_initializations": 0,
            "physics_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
        },
    )


def _check_inputs(inputs):
    for key, value in inputs.items():
        if key.endswith("_ref"):
            prefix.checked(value)
    for item in inputs["cases"]:
        for ref in item["refs"].values():
            prefix.checked(ref)


def execute_phase(phase, worker_module):
    """Own all 30 baseline workers and the one studied-state worker."""
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "Registered shadow phase")
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Complete source-frozen CPU QA",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "QA source identity")
    inputs, sources = load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True and dry["inputs"] == inputs and dry["source_refs"] == sources,
        "Zero-model dry identity",
    )
    producer_refs = None
    if phase == "checker":
        prior = prefix.read(prefix.reference(ROOT / "producer/result.json"))
        registration = prefix.read(prior["registration_ref"])
        prefix.require(
            prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True
            and registration["inputs"] == inputs
            and registration["source_refs"] == sources
            and len(prior["case_refs"]) == 30,
            "Closed producer and unchanged sources",
        )
        producer_refs = {"cases": prior["case_refs"], "studied": prior["studied_ref"]}
    directory = ROOT / phase
    directory.mkdir(parents=True, exist_ok=False)
    registration_ref = publish(
        directory / "registration.json",
        {"phase": phase, "inputs": inputs, "source_refs": sources, "producer": producer_refs},
    )
    keys = [f"case_{i:02d}" for i in range(30)] + ["studied"]
    specs = [
        {
            "key": key,
            "argv": [sys.executable, "-m", worker_module, "--worker", str(index)],
            "log": directory / key / "worker.log",
        }
        for index, key in enumerate(keys)
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=600)
    for worker in lifecycle["workers"]:
        worker["log_ref"] = prefix.reference(worker.pop("log"))
    workers_ref = publish(directory / "workers.json", lifecycle)
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "phase": phase,
        "registration_ref": registration_ref,
        "workers_ref": workers_ref,
        "model_initializations": 0,
        "physics_steps": 0,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and [item["key"] for item in lifecycle["workers"]] == keys
            and all(
                item["returncode"] == 0 and item["actual_wait"] and item["reaped"] and not item["signal_sent"]
                for item in lifecycle["workers"]
            ),
            "All thirty-one independent workers exited and were reaped",
        )
        refs = [prefix.reference(directory / key / "result.json") for key in keys]
        cases, studied = [prefix.read(ref) for ref in refs[:30]], prefix.read(refs[30])
        prefix.require(
            all(item["complete"] is item["passed"] is True for item in cases + [studied])
            and [item["case_id"] for item in cases] == [item["case_id"] for item in inputs["cases"]],
            "All original case results in registered order",
        )
        groups = {}
        for family in ("correct_reference", "development"):
            selected = [item["summary"] for item in cases if item["family"] == family]
            groups[family] = {
                "cases": len(selected),
                **{
                    key: sum(item[key] for item in selected)
                    for key in (
                        "baseline_periods",
                        "queries",
                        "conflicts_without_safety",
                        "unknown_inner_conflicts",
                        "holds",
                        "bridges",
                    )
                },
            }
        prefix.require(
            groups["correct_reference"]["cases"] == 18
            and groups["correct_reference"]["baseline_periods"] == 4907
            and groups["development"]["cases"] == 12
            and groups["development"]["baseline_periods"] == 7200
            and groups["development"]["queries"] == 900,
            "Full original matrix denominator",
        )
        _check_inputs(inputs)
        prefix.require(source_refs() == sources and snapshot(REPO)["sources"] == before, "Source and input identity")
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            case_refs=refs[:30],
            studied_ref=refs[30],
            summary=groups,
            studied_summary=studied["summary"],
        )
    finally:
        print(publish(directory / "result.json", result), flush=True)
