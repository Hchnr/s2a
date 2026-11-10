"""Independently check all saved replan events and matched reference endpoints."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/replan_task_event_stability_check_20261110.md"
QA = REPO / "outputs/qa/replan_task_event_stability_check_20261110/result.json"
DRY = REPO / "outputs/replan_task_event_stability_check_dry_20261110/result.json"
OUTPUT = REPO / "outputs/replan_task_event_stability_check_20261110"
PRODUCER = {
    "path": str(REPO / "outputs/replan_task_event_stability_20261110/result.json"),
    "sha256": "9cfe6d4e06512ea79026ea531ec12d030d29e868a0a77fe8ed9569a132d28272",
}


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            Path(__file__),
            PLAN,
            REPO / "tests/research/wam/tracking/preview/test_replan_task_event_independent_check.py",
        )
    ]


def load_inputs():
    original = prefix.read(PRODUCER)
    prefix.require(
        original["complete"] is original["passed"] is original["sources_unchanged"] is True
        and len(original["case_refs"]) == 12
        and original["summary"]["matched_endpoints"] == 28416,
        "One closed twelve-case producer",
    )
    registration = prefix.read(original["registration_ref"])
    prefix.require(len(registration["cases"]) == 12, "Original twelve registered source cases")
    cases = []
    for index, (item, result_ref) in enumerate(zip(registration["cases"], original["case_refs"], strict=True)):
        result = prefix.read(result_ref)
        prefix.require(
            result["complete"] is result["passed"] is True
            and result["data"]["case"]["case_index"] == item["case"]["case_index"] == index
            and result["input_refs"] == item["refs"],
            "Original result and input case identities",
        )
        for ref in item["refs"].values():
            prefix.checked(ref)
        cases.append({"case": item["case"], "target": item["target"], "refs": item["refs"], "result_ref": result_ref})
    return {"producer_ref": PRODUCER, "original_registration_ref": original["registration_ref"], "cases": cases}


def magnitude_error(a, b):
    """Use body-frame relative rotation, independently of producer pose_error."""
    position = math.dist(np.asarray(a)[:3, 3], np.asarray(b)[:3, 3])
    relative = np.asarray(a)[:3, :3].T @ np.asarray(b)[:3, :3]
    return position, float(Rotation.from_matrix(relative).magnitude())


def assert_near(actual, expected):
    if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual - expected) > 1e-10:
        raise ValueError("Independent numerical reference reconstruction mismatch")


def verify_case(item, produced, chunks, rows, tool):
    prefix.require(len(chunks) == 75 and len(rows) == 600 and len(produced["pairs"]) == 74, "Complete case timeline")
    prefix.require(produced["case"] == item["case"] and produced["target"] == item["target"], "Original case identity")
    target = item["target"]
    tool = np.asarray(tool, dtype=float)
    grasp, success = 0, 0
    for query, chunk in enumerate(chunks):
        step = 8 * query
        prefix.require(chunk["step"] == step and len(chunk["openings"]) == 40, "Original generated clock")
        for offset in range(8):
            row = rows[step + offset]
            prefix.require(
                row["step"] == step + offset
                and row["chunk_index"] == query
                and row["chunk_physical_step"] == offset
                and row["opening_target"] == chunk["openings"][offset],
                "Original executed reference clock and opening",
            )
            prefix.require(
                type(row["success"]) is bool and type(row["objects"][target]["grasp_contact"]) is bool,
                "Native booleans",
            )
            grasp += row["objects"][target]["grasp_contact"]
            success += row["success"]
        anchor = np.asarray(rows[step]["measured_source"]) @ tool
        np.testing.assert_allclose(anchor, chunk["anchor_tcp"], atol=1e-10, rtol=0)
        assert_near(float(rows[step]["opening_before"]), float(chunk["anchor_opening"]))
    prefix.require(
        produced["grasp_contact_steps"] == grasp
        and produced["task_success_steps"] == success
        and produced["queries"] == 75
        and produced["periods"] == 600
        and produced["reference_queries_with_closure"]
        == sum(any(value < 0.35 for value in chunk["openings"]) for chunk in chunks)
        and produced["reference_queries_with_opening"]
        == sum(any(value > 0.8 for value in chunk["openings"]) for chunk in chunks)
        and produced["executed_opening_targets_below_035"] == sum(row["opening_target"] < 0.35 for row in rows)
        and produced["executed_opening_targets_above_08"] == sum(row["opening_target"] > 0.8 for row in rows),
        "Original physical outcome and opening counts",
    )

    summary = {
        name: {
            "deferred": 0,
            "revisions": dict.fromkeys(("earlier", "same", "later", "not_in_new_horizon"), 0),
            "opposite_opening_at_old_event": 0,
            "event_pose_agreement": 0,
            "next_prefix_and_current_agreement": 0,
            "current_opposite_phase": 0,
            "current_opposite_and_event_pose_agreement": 0,
            "current_opposite_and_prefix_current_agreement": 0,
            "opportunities_next_eight": 0,
            "opportunity_event_pose": 0,
            "opportunity_prefix_current": 0,
            "opportunity_all": 0,
        }
        for name in ("closure", "opening")
    }
    maximum = {"position": 0.0, "rotation": 0.0, "opening": 0.0}
    for query, saved in enumerate(produced["pairs"], start=1):
        old, new = chunks[query - 1], chunks[query]
        old_p, new_p = old["effective_targets_tcp"], new["effective_targets_tcp"]
        old_o, new_o = old["openings"], new["openings"]
        old_step, new_step = (query - 1) * 8, query * 8
        prefix.require(saved["old_step"] == old_step and saved["new_step"] == new_step, "Replan pair timing")
        positions, rotations = [], []
        close_to_open, open_to_close = 0, 0
        for new_index in range(32):
            old_index = 8 + new_index
            position, rotation = magnitude_error(old_p[old_index], new_p[new_index])
            difference = float(old_o[old_index]) - float(new_o[new_index])
            for key, value, label in (
                ("position_difference_m", position, "position"),
                ("rotation_difference_rad", rotation, "rotation"),
                ("opening_difference", difference, "opening"),
            ):
                delta = abs(value - saved[key][new_index])
                assert_near(value, saved[key][new_index])
                maximum[label] = max(maximum[label], delta)
            positions.append(position)
            rotations.append(rotation)
            close_to_open += old_o[old_index] < 0.35 and new_o[new_index] > 0.8
            open_to_close += old_o[old_index] > 0.8 and new_o[new_index] < 0.35
        prefix.require(
            saved["old_closed_new_open"] == close_to_open and saved["old_open_new_closed"] == open_to_close,
            "Matched opening phase contradictions",
        )
        current = np.asarray(rows[new_step]["measured_source"]) @ tool
        current_position, current_rotation = magnitude_error(old_p[8], current)
        assert_near(current_position, saved["current_position_error_m"])
        assert_near(current_rotation, saved["current_rotation_error_rad"])
        assert_near(float(rows[new_step]["opening_before"]), saved["current_opening"])
        prefix_ok = max(positions[:8]) <= 0.02 and max(rotations[:8]) <= 0.1
        current_ok = current_position <= 0.03 and current_rotation <= 0.15
        prefix.require(
            saved["next_prefix_pose_agreement"] is prefix_ok and saved["current_near_old_prefix"] is current_ok,
            "Independent prefix and current pose scales",
        )
        for name in ("closure", "opening"):
            threshold = (lambda value: value < 0.35) if name == "closure" else (lambda value: value > 0.8)
            old_first = next((i for i, value in enumerate(old_o) if threshold(value)), None)
            new_first = next((i for i, value in enumerate(new_o) if threshold(value)), None)
            event = saved[name]
            deferred = old_first is not None and old_first >= 8
            prefix.require(
                event["old_first_index"] == old_first
                and event["new_first_index"] == new_first
                and event["deferred_in_old"] is deferred,
                "Independent first-event timing",
            )
            if not deferred:
                continue
            same_time = old_first - 8
            shift = None if new_first is None else new_first - same_time
            category = (
                "not_in_new_horizon" if shift is None else "earlier" if shift < 0 else "later" if shift > 0 else "same"
            )
            current_opening = float(rows[new_step]["opening_before"])
            current_opposite = current_opening > 0.8 if name == "closure" else current_opening < 0.35
            target_opposite = bool(new_o[same_time] > 0.8 if name == "closure" else new_o[same_time] < 0.35)
            event_pose = positions[same_time] <= 0.02 and rotations[same_time] <= 0.1
            prefix.require(
                event["old_absolute_endpoint"] == old_step + old_first + 1
                and event["new_first_absolute_endpoint"] == (None if new_first is None else new_step + new_first + 1)
                and event["new_first_shift_steps"] == shift
                and event["first_event_revision"] == category
                and event["new_opening_at_old_event"] == new_o[same_time]
                and event["opposite_opening_at_old_event"] is target_opposite
                and event["event_pose_agreement"] is event_pose
                and event["next_prefix_pose_agreement"] is prefix_ok
                and event["current_near_old_prefix"] is current_ok
                and event["currently_opposite_phase"] is current_opposite,
                "Independent event revision and pose agreement",
            )
            assert_near(positions[same_time], event["event_position_difference_m"])
            assert_near(rotations[same_time], event["event_rotation_difference_rad"])
            counts = summary[name]
            counts["deferred"] += 1
            counts["revisions"][category] += 1
            counts["opposite_opening_at_old_event"] += target_opposite
            counts["event_pose_agreement"] += event_pose
            counts["next_prefix_and_current_agreement"] += prefix_ok and current_ok
            counts["current_opposite_phase"] += current_opposite
            counts["current_opposite_and_event_pose_agreement"] += current_opposite and event_pose
            counts["current_opposite_and_prefix_current_agreement"] += current_opposite and prefix_ok and current_ok
            if old_first < 16 and current_opposite and category in ("later", "not_in_new_horizon"):
                counts["opportunities_next_eight"] += 1
                counts["opportunity_event_pose"] += event_pose
                counts["opportunity_prefix_current"] += prefix_ok and current_ok
                counts["opportunity_all"] += event_pose and prefix_ok and current_ok
    return {"pairs": 74, "matched_endpoints": 74 * 32, "max_differences": maximum, "events": summary}


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    item = registration["cases"][index]
    prefix.require(item["case"]["case_index"] == index, "Original worker identity")
    produced = prefix.read(item["result_ref"])["data"]
    refs = item["refs"]
    chunks = prefix.read(refs["chunks"])
    tool = prefix.read(refs["tool"])["source_from_tcp"]
    with prefix.checked(refs["steps"]).open() as stream:
        rows = [json.loads(line) for line in stream]
    data = verify_case(item, produced, chunks, rows, tool)
    for ref in (*refs.values(), item["result_ref"]):
        prefix.checked(ref)
    print(
        publish(OUTPUT / f"case_{index:02d}" / "result.json", {"complete": True, "passed": True, "data": data}),
        flush=True,
    )


def verify_summary(original, checked, cases):
    """Recompute the producer's aggregate on the already checked endpoints."""
    summary = original["summary"]
    pairs = [pair for case in cases for pair in case["pairs"]]
    prefix.require(
        summary["cases"] == len(cases) == 12
        and summary["queries"] == sum(case["queries"] for case in cases) == 900
        and summary["periods"] == sum(case["periods"] for case in cases) == 7200
        and summary["replan_pairs"] == len(pairs) == 888
        and summary["matched_endpoints"] == 28416,
        "Independent complete aggregate denominators",
    )

    def quantile(values, fraction):
        ordered = sorted(values)
        index = (len(ordered) - 1) * fraction
        left, right = math.floor(index), math.ceil(index)
        return ordered[left] + (ordered[right] - ordered[left]) * (index - left)

    for field, source, absolute in (
        ("position_difference_m", "position_difference_m", False),
        ("rotation_difference_rad", "rotation_difference_rad", False),
        ("absolute_opening_difference", "opening_difference", True),
    ):
        values = [abs(value) if absolute else value for pair in pairs for value in pair[source]]
        prefix.require(len(values) == 28416, "Complete aggregate endpoint values")
        assert_near(summary[field]["median"], quantile(values, 0.5))
        assert_near(summary[field]["p90"], quantile(values, 0.9))
        assert_near(summary[field]["max"], max(values))
    for name in ("old_closed_new_open", "old_open_new_closed", "next_prefix_pose_agreement"):
        prefix.require(summary[name] == sum(pair[name] for pair in pairs), "Independent aggregate pair counts")
    prefix.require(
        summary["next_prefix_and_current_agreement"]
        == sum(pair["next_prefix_pose_agreement"] and pair["current_near_old_prefix"] for pair in pairs),
        "Independent aggregate current and prefix count",
    )
    for name in ("closure", "opening"):
        actual = [case["events"][name] for case in checked]
        for key in (
            "deferred",
            "opposite_opening_at_old_event",
            "event_pose_agreement",
            "next_prefix_and_current_agreement",
            "current_opposite_phase",
            "current_opposite_and_event_pose_agreement",
            "current_opposite_and_prefix_current_agreement",
        ):
            prefix.require(sum(row[key] for row in actual) == summary[name][key], "Independent event totals")
        for category in ("earlier", "same", "later", "not_in_new_horizon"):
            prefix.require(
                sum(row["revisions"][category] for row in actual) == summary[name]["revisions"][category],
                "Independent first-event revision totals",
            )


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
    prefix.require(dry["inputs"] == inputs and dry["source_refs"] == sources and dry["passed"] is True, "Dry identity")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration_ref = publish(OUTPUT / "registration.json", {**inputs, "source_refs": sources})
    specs = [
        {
            "key": f"case_{index:02d}",
            "argv": [
                sys.executable,
                "-m",
                "scripts.verification.reference_ports.check_replan_task_events",
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
            "All independent workers closed and were reaped",
        )
        refs = [prefix.reference(OUTPUT / f"case_{index:02d}" / "result.json") for index in range(12)]
        cases = [prefix.read(ref)["data"] for ref in refs]
        prefix.require(
            sum(case["pairs"] for case in cases) == 888 and sum(case["matched_endpoints"] for case in cases) == 28416,
            "All original endpoints independently checked",
        )
        original = prefix.read(PRODUCER)
        original_cases = [prefix.read(item["result_ref"])["data"] for item in inputs["cases"]]
        verify_summary(original, cases, original_cases)
        for item in inputs["cases"]:
            for ref in (*item["refs"].values(), item["result_ref"]):
                prefix.checked(ref)
        prefix.require(source_refs() == sources and snapshot(REPO)["sources"] == before, "Sources and inputs unchanged")
        result.update(
            complete=True, passed=True, sources_unchanged=True, case_refs=refs, pairs=888, matched_endpoints=28416
        )
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
                    "inputs": load_inputs(),
                    "source_refs": source_refs(),
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
