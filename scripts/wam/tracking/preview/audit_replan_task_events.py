"""Describe replan phase revisions on all independently reviewed v4 cases."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.tracking.preview.replan_task_events import compare_replans
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/replan_task_event_stability_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/replan_task_event_stability_20261110.md"
QA = REPO / "outputs/qa/replan_task_event_stability_20261110/result.json"
DRY = REPO / "outputs/replan_task_event_stability_dry_20261110/result.json"
OUTPUT = REPO / "outputs/replan_task_event_stability_20261110"
REVIEW = prefix.REVIEW
TARGETS = ("akita_black_bowl_1", "wine_bottle_1", "cream_cheese_1")
CASE_BASE = REPO / "outputs/tracking_preview_wam_development_v4_20261027/development/cases"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/wam/tracking/preview/replan_task_events.py",
            REPO / "tests/research/wam/tracking/preview/test_replan_task_events.py",
            REPO / "tests/research/wam/tracking/preview/test_replan_task_event_inputs.py",
        )
    ]


def expected_case(index):
    prefix.require(type(index) is int and 0 <= index < 12, "One of the original twelve cases")
    task, within = divmod(index, 4)
    seed, variant = (42, 43)[within // 2], ("raw", "terminal058")[within % 2]
    return f"task{task}_init0_s{seed}_{variant}"


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        config
        == {
            "schema": "s2a.replan-task-event-stability.v1",
            "cases": list(range(12)),
            "steps_per_case": 600,
            "queries_per_case": 75,
            "query_interval": 8,
            "horizon": 40,
            "closing_below": 0.35,
            "opening_above": 0.8,
            "reference_position_scale_m": 0.02,
            "reference_rotation_scale_rad": 0.1,
            "current_position_scale_m": 0.03,
            "current_rotation_scale_rad": 0.15,
            "max_workers": 12,
            "worker_timeout_seconds": 600,
            "model_initializations": 0,
            "physics_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered twelve-case descriptive audit",
    )
    review = prefix.read(REVIEW)
    prefix.require(
        review["complete"] is review["passed"] is True and len(review["cases"]) == 12,
        "Twelve independently reviewed development cases required",
    )
    source_map, cases = review["source_and_artifact_sha256"], []

    def verified(path):
        ref = prefix.reference(path)
        prefix.require(source_map.get(ref["path"]) == ref["sha256"], "Reviewed development input identity")
        return ref

    for index, item in enumerate(review["cases"]):
        case = item["case"]
        prefix.require(
            case["case_index"] == index
            and case["case_id"] == expected_case(index)
            and item["mechanical"]["steps"] == 600
            and item["mechanical"]["chunks"] == 75,
            "Original case order and full physical budget",
        )
        folder = Path(item["mechanical"]["case"])
        prefix.require(folder.is_relative_to(CASE_BASE / expected_case(index)), "Original case folder")
        manifest_ref = verified(folder / "manifest.json")
        manifest = prefix.read(manifest_ref)
        refs = {
            "manifest": manifest_ref,
            "chunks": verified(folder / "chunks.json"),
            "steps": verified(folder / "steps.jsonl"),
            "tool": verified(Path(manifest["runtime_config"]["tool_contract"])),
        }
        cases.append({"case": case, "target": TARGETS[index // 4], "refs": refs})
    return {"config_ref": config_ref, "review_ref": REVIEW, "cases": cases}


def describe_case(case, target, chunks, rows, tool):
    """Check original causal timing, then describe overlapping generated points."""
    prefix.require(len(chunks) == 75 and len(rows) == 600, "Complete original chunk and control sequences")
    tool = np.asarray(tool, dtype=float)
    prefix.require(tool.shape == (4, 4) and np.isfinite(tool).all(), "Finite original TCP transform")
    decoded = []
    for query, chunk in enumerate(chunks):
        step = query * 8
        prefix.require(chunk["step"] == step, "Original replan timing")
        pose, opening = (
            np.asarray(chunk["effective_targets_tcp"], dtype=float),
            np.asarray(chunk["openings"], dtype=float),
        )
        prefix.require(pose.shape == (40, 4, 4) and opening.shape == (40,), "Complete generated reference")
        for offset in range(8):
            row = rows[step + offset]
            prefix.require(
                row["step"] == step + offset
                and row["chunk_index"] == query
                and row["chunk_physical_step"] == offset
                and row["opening_target"] == float(opening[offset]),
                "Each executed opening retains its original chunk and endpoint",
            )
            prefix.require(
                type(row["success"]) is bool and type(row["objects"][target]["grasp_contact"]) is bool,
                "Original native task and grasp descriptors",
            )
        anchor = np.asarray(chunk["anchor_tcp"], dtype=float)
        actual = np.asarray(rows[step]["measured_source"], dtype=float) @ tool
        np.testing.assert_allclose(anchor, actual, atol=1e-10, rtol=0)
        np.testing.assert_allclose(chunk["anchor_opening"], rows[step]["opening_before"], atol=1e-12, rtol=0)
        decoded.append((pose, opening, anchor, float(rows[step]["opening_before"])))

    pairs = []
    for query in range(1, 75):
        old, old_o, _, _ = decoded[query - 1]
        new, new_o, current, opening = decoded[query]
        pairs.append(
            compare_replans(
                old,
                old_o,
                new,
                new_o,
                old_step=(query - 1) * 8,
                new_step=query * 8,
                measured_tcp=current,
                opening=opening,
            )
        )
    return {
        "case": case,
        "target": target,
        "queries": len(chunks),
        "periods": len(rows),
        "pairs": pairs,
        "reference_queries_with_closure": sum(bool(np.any(item[1] < 0.35)) for item in decoded),
        "reference_queries_with_opening": sum(bool(np.any(item[1] > 0.8)) for item in decoded),
        "executed_opening_targets_below_035": sum(row["opening_target"] < 0.35 for row in rows),
        "executed_opening_targets_above_08": sum(row["opening_target"] > 0.8 for row in rows),
        "grasp_contact_steps": sum(row["objects"][target]["grasp_contact"] for row in rows),
        "task_success_steps": sum(row["success"] for row in rows),
    }


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    item = registration["cases"][index]
    prefix.require(item["case"]["case_id"] == expected_case(index), "Original worker identity")
    refs = item["refs"]
    chunks = prefix.read(refs["chunks"])
    tool = prefix.read(refs["tool"])["source_from_tcp"]
    path = prefix.checked(refs["steps"])
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    data = describe_case(item["case"], item["target"], chunks, rows, tool)
    for ref in refs.values():
        prefix.checked(ref)
    result = {"complete": True, "passed": True, "input_refs": refs, "data": data}
    print(publish(OUTPUT / f"case_{index:02d}" / "result.json", result), flush=True)


def summarize(cases):
    pairs = [pair for case in cases for pair in case["pairs"]]
    prefix.require(len(cases) == 12 and len(pairs) == 888, "Complete paired development matrix")
    positions = np.asarray([v for pair in pairs for v in pair["position_difference_m"]])
    rotations = np.asarray([v for pair in pairs for v in pair["rotation_difference_rad"]])
    opening = np.asarray([abs(v) for pair in pairs for v in pair["opening_difference"]])
    prefix.require(positions.shape == rotations.shape == opening.shape == (28416,), "All matched endpoints")

    def stats(values):
        return {
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "max": float(np.max(values)),
        }

    def events(direction):
        deferred = [pair[direction] for pair in pairs if pair[direction]["deferred_in_old"]]
        current_opposite = [event for event in deferred if event["currently_opposite_phase"]]
        return {
            "deferred": len(deferred),
            "revisions": {
                label: sum(event["first_event_revision"] == label for event in deferred)
                for label in ("earlier", "same", "later", "not_in_new_horizon")
            },
            "opposite_opening_at_old_event": sum(event["opposite_opening_at_old_event"] for event in deferred),
            "event_pose_agreement": sum(event["event_pose_agreement"] for event in deferred),
            "next_prefix_and_current_agreement": sum(
                event["next_prefix_pose_agreement"] and event["current_near_old_prefix"] for event in deferred
            ),
            "current_opposite_phase": len(current_opposite),
            "current_opposite_and_event_pose_agreement": sum(
                event["event_pose_agreement"] for event in current_opposite
            ),
            "current_opposite_and_prefix_current_agreement": sum(
                event["next_prefix_pose_agreement"] and event["current_near_old_prefix"] for event in current_opposite
            ),
        }

    return {
        "cases": len(cases),
        "queries": sum(case["queries"] for case in cases),
        "periods": sum(case["periods"] for case in cases),
        "replan_pairs": len(pairs),
        "matched_endpoints": len(positions),
        "position_difference_m": stats(positions),
        "rotation_difference_rad": stats(rotations),
        "absolute_opening_difference": stats(opening),
        "old_closed_new_open": sum(pair["old_closed_new_open"] for pair in pairs),
        "old_open_new_closed": sum(pair["old_open_new_closed"] for pair in pairs),
        "next_prefix_pose_agreement": sum(pair["next_prefix_pose_agreement"] for pair in pairs),
        "next_prefix_and_current_agreement": sum(
            pair["next_prefix_pose_agreement"] and pair["current_near_old_prefix"] for pair in pairs
        ),
        "closure": events("closure"),
        "opening": events("opening"),
    }


def execute():
    prefix.environment()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Frozen CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU QA source identity")
    inputs, sources = load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True and dry["inputs"] == inputs and dry["source_refs"] == sources,
        "Zero-model dry preserves registered sources and inputs",
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration_ref = publish(OUTPUT / "registration.json", {**inputs, "source_refs": sources})
    specs = [
        {
            "key": f"case_{index:02d}",
            "argv": [
                sys.executable,
                "-m",
                "scripts.wam.tracking.preview.audit_replan_task_events",
                "--worker",
                str(index),
            ],
            "log": OUTPUT / f"case_{index:02d}" / "worker.log",
        }
        for index in range(12)
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=600)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers_ref = publish(OUTPUT / "workers.json", lifecycle)
    result = {
        "complete": False,
        "passed": False,
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
            and len(lifecycle["workers"]) == 12
            and all(
                item["returncode"] == 0 and item["actual_wait"] and item["reaped"] and not item["signal_sent"]
                for item in lifecycle["workers"]
            ),
            "All twelve worker processes closed and were reaped",
        )
        refs = [prefix.reference(OUTPUT / f"case_{index:02d}" / "result.json") for index in range(12)]
        results = [prefix.read(ref) for ref in refs]
        prefix.require(
            all(item["complete"] is item["passed"] is True for item in results)
            and [item["data"]["case"]["case_index"] for item in results] == list(range(12)),
            "Twelve original cases completed in registered order",
        )
        for item in inputs["cases"]:
            for ref in item["refs"].values():
                prefix.checked(ref)
        prefix.require(source_refs() == sources and snapshot(REPO)["sources"] == before, "Unchanged source and inputs")
        data = [item["data"] for item in results]
        result.update(complete=True, passed=True, sources_unchanged=True, case_refs=refs, summary=summarize(data))
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", type=int)
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        print(
            publish(
                DRY,
                {
                    "complete": True,
                    "passed": True,
                    "source_refs": source_refs(),
                    "inputs": load_inputs(),
                    "model_initializations": 0,
                    "physics_steps": 0,
                    "world_model_calls": 0,
                    "fresh_cases": 0,
                },
            ),
            flush=True,
        )
    elif args.run:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
