"""Independently reconstruct each saved causal event-memory shadow decision."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/causal_event_memory_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/causal_event_memory_check_20261110.md"
QA = REPO / "outputs/qa/causal_event_memory_check_20261110/result.json"
DRY = REPO / "outputs/causal_event_memory_check_dry_20261110/result.json"
OUTPUT = REPO / "outputs/causal_event_memory_check_20261110"
PRODUCER = {
    "path": str(REPO / "outputs/causal_event_memory_20261110/result.json"),
    "sha256": "da654fd4505d2afdb12460b3bc30ce53882a7818b523eb5e85090821cfa8e986",
}
CASE_BASE = REPO / "outputs/tracking_preview_wam_development_v4_20261027/development/cases"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            Path(__file__),
            PLAN,
            CONFIG,
            REPO / "tests/research/wam/tracking/preview/test_causal_event_memory_independent_check.py",
        )
    ]


def expected_case(index):
    prefix.require(type(index) is int and 0 <= index < 12, "One reviewed development case")
    task, within = divmod(index, 4)
    return f"task{task}_init0_s{(42, 43)[within // 2]}_{('raw', 'terminal058')[within % 2]}"


def load_inputs():
    produced = prefix.read(PRODUCER)
    prefix.require(
        produced["complete"] is produced["passed"] is produced["sources_unchanged"] is True
        and produced["summary"]["cases"] == 12
        and produced["summary"]["queries"] == 900
        and produced["summary"]["baseline_periods"] == 7200
        and len(produced["case_refs"]) == 12,
        "Closed registered shadow producer",
    )
    registration = prefix.read(produced["registration_ref"])
    prefix.require(
        registration["config_ref"] == prefix.reference(CONFIG) and registration["review_ref"] == prefix.REVIEW,
        "Original shadow configuration and reviewed-source identity",
    )
    config = prefix.read(registration["config_ref"])
    prefix.require(
        config
        == {
            "schema": "s2a.causal-event-memory.v1",
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
        "Exact fixed shadow protocol",
    )
    review = prefix.read(prefix.REVIEW)
    prefix.require(review["complete"] is review["passed"] is True and len(review["cases"]) == 12, "Original review")
    source_map, cases = review["source_and_artifact_sha256"], []
    for index, (item, result_ref) in enumerate(zip(registration["cases"], produced["case_refs"], strict=True)):
        case, refs = item["case"], item["refs"]
        result = prefix.read(result_ref)
        prefix.require(
            case == review["cases"][index]["case"]
            and case["case_index"] == index
            and case["case_id"] == expected_case(index)
            and result["complete"] is result["passed"] is True
            and result["input_refs"] == refs
            and result["data"]["case"] == case,
            "Original ordered development case and producer receipt",
        )
        prefix.require(
            Path(refs["chunks"]["path"]).is_relative_to(CASE_BASE / case["case_id"]),
            "Development-case input; final held-out cases excluded",
        )
        for ref in refs.values():
            prefix.checked(ref)
            prefix.require(source_map.get(ref["path"]) == ref["sha256"], "Reviewed historical input SHA")
        cases.append({"case": case, "refs": refs, "result_ref": result_ref})
    return {
        "producer_ref": PRODUCER,
        "registration_ref": produced["registration_ref"],
        "config_ref": registration["config_ref"],
        "cases": cases,
    }


def distance(a, b):
    """Measure translation and body-relative SO(3) angle without producer geometry."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    prefix.require(a.shape == b.shape == (4, 4) and np.isfinite(a).all() and np.isfinite(b).all(), "Finite pose pair")
    return math.dist(a[:3, 3], b[:3, 3]), float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())


def close(actual, expected, maximum):
    delta = abs(float(actual) - float(expected))
    prefix.require(math.isfinite(delta) and delta <= 1e-10, "Independent numerical reconstruction mismatch")
    return max(maximum, delta)


def first_phase(openings, phase, close_limit, open_limit):
    if phase is None:
        return None
    for index, opening in enumerate(openings):
        if (phase == "closure" and opening < close_limit) or (phase == "opening" and opening > open_limit):
            return index
    return None


def state(last, pending):
    return {"last_completed_replan_step": last, "faulted": False, "staged": False, "pending_event": pending}


def verify_case(case, produced, chunks, rows, tool, config):
    """Replay only original observations, keeping an independent event description."""
    stride, horizon = config["query_interval"], config["horizon"]
    prefix.require(
        produced["case"] == case
        and produced["queries"] == len(chunks) == 75
        and produced["baseline_periods"] == len(rows) == 600
        and len(produced["records"]) == 75,
        "Complete original timeline and shadow receipt",
    )
    tool = np.asarray(tool, dtype=float)
    prefix.require(tool.shape == (4, 4) and np.isfinite(tool).all(), "Original TCP transform")
    pending, last, first_change = None, None, None
    reasons, max_difference = Counter(), 0.0
    armed_count = retained_count = closure_count = opening_count = changed_count = later_count = 0
    for query, (chunk, saved) in enumerate(zip(chunks, produced["records"], strict=True)):
        step = query * stride
        poses = np.asarray(chunk["effective_targets_tcp"], dtype=float)
        openings = np.asarray(chunk["openings"], dtype=float)
        prefix.require(
            chunk["step"] == step
            and poses.shape == (horizon, 4, 4)
            and openings.shape == (horizon,)
            and np.isfinite(poses).all()
            and np.isfinite(openings).all()
            and np.all((0 <= openings) & (openings <= 1)),
            "Complete finite generated reference at the original clock",
        )
        for offset in range(stride):
            row = rows[step + offset]
            prefix.require(
                row["step"] == step + offset
                and row["chunk_index"] == query
                and row["chunk_physical_step"] == offset
                and row["opening_target"] == float(openings[offset]),
                "Original executed opening and time",
            )
        current = np.asarray(rows[step]["measured_source"], dtype=float) @ tool
        current_opening = float(rows[step]["opening_before"])
        np.testing.assert_allclose(current, chunk["anchor_tcp"], atol=1e-10, rtol=0)
        max_difference = close(current_opening, chunk["anchor_opening"], max_difference)
        prefix.require(saved["memory_before"] == state(last, pending), "Prior committed memory")
        phase = (
            "closure"
            if current_opening > config["opening_above"]
            else "opening"
            if current_opening < config["closing_below"]
            else None
        )
        old, next_pending = pending, None
        reason, source_step, guards, activated = "no_pending_event", step, None, False
        if old is not None:
            offset = step - old["source_step"]
            remaining = old["event_index"] - offset
            if phase != old["direction"]:
                reason = "measured_phase_changed"
            elif offset <= 0 or offset + stride > horizon or remaining < 0:
                reason = "event_expired"
            else:
                newest = first_phase(openings, phase, config["closing_below"], config["opening_above"])
                if newest is not None and newest <= remaining:
                    reason = "latest_covers_event"
                else:
                    old_poses = np.asarray(chunks[old["source_step"] // stride]["effective_targets_tcp"], dtype=float)
                    prefix_errors = [distance(old_poses[offset + i], poses[i]) for i in range(stride)]
                    event_error = distance(old_poses[old["event_index"]], poses[remaining])
                    current_error = distance(old_poses[offset], current)
                    guards = {
                        "prefix_position_m": max(value[0] for value in prefix_errors),
                        "prefix_rotation_rad": max(value[1] for value in prefix_errors),
                        "event_position_m": event_error[0],
                        "event_rotation_rad": event_error[1],
                        "current_position_m": current_error[0],
                        "current_rotation_rad": current_error[1],
                    }
                    agreement = (
                        guards["prefix_position_m"] <= config["reference_position_scale_m"]
                        and guards["prefix_rotation_rad"] <= config["reference_rotation_scale_rad"]
                        and guards["event_position_m"] <= config["reference_position_scale_m"]
                        and guards["event_rotation_rad"] <= config["reference_rotation_scale_rad"]
                        and guards["current_position_m"] <= config["current_position_scale_m"]
                        and guards["current_rotation_rad"] <= config["current_rotation_scale_rad"]
                    )
                    if not agreement:
                        reason = "pose_agreement_rejected"
                    elif remaining >= stride:
                        reason, next_pending = "event_retained", old
                    else:
                        reason, source_step, activated = "event_due", old["source_step"], True
        armed = False
        if next_pending is None and not activated:
            event_index = first_phase(openings, phase, config["closing_below"], config["opening_above"])
            if event_index is not None and event_index >= stride:
                next_pending = {
                    "direction": phase,
                    "source_step": step,
                    "event_index": event_index,
                    "absolute_endpoint": step + event_index + 1,
                }
                armed = True
        if guards is None:
            prefix.require(saved["guards"] is None, "No uncomputed pose guards")
        else:
            prefix.require(set(saved["guards"]) == set(guards), "All six independently computed pose guards")
            for key in guards:
                max_difference = close(saved["guards"][key], guards[key], max_difference)
        indices = list(range(step - source_step, step - source_step + stride))
        source = chunks[source_step // stride]
        selected_poses = np.asarray(source["effective_targets_tcp"], dtype=float)[indices]
        selected_openings = np.asarray(source["openings"], dtype=float)[indices]
        np.testing.assert_array_equal(saved["poses"], selected_poses)
        np.testing.assert_array_equal(saved["openings"], selected_openings)
        changed = sum(float(a) != float(b) for a, b in zip(selected_openings, openings[:stride], strict=True))
        if activated:
            prefix.require(changed > 0 and old is not None, "An intervention must use an earlier event")
            closure_count += old["direction"] == "closure"
            opening_count += old["direction"] == "opening"
            changed_count += changed
            if first_change is None:
                first_change, label = step, "first_intervention_on_baseline"
            else:
                label = "baseline_observation_after_intervention"
        elif first_change is None:
            label = "baseline_observation_before_intervention"
        else:
            label = "baseline_observation_after_intervention"
        later_count += label == "baseline_observation_after_intervention"
        expected = {
            "query": query,
            "step": step,
            "source": "remembered" if activated else "latest",
            "source_step": source_step,
            "source_indices": indices,
            "absolute_endpoints": [step + i + 1 for i in range(stride)],
            "reason": reason,
            "armed_new_event": armed,
            "event": old if old is not None else next_pending,
            "baseline_observation_label": label,
            "changed_opening_endpoints": changed,
            "memory_after": state(step, next_pending),
        }
        for key, value in expected.items():
            prefix.require(saved[key] == value, f"Independent decision mismatch: {key}")
        chosen_errors = [distance(selected_poses[i], poses[i]) for i in range(stride)]
        max_difference = close(
            saved["max_selected_vs_latest_position_m"], max(value[0] for value in chosen_errors), max_difference
        )
        max_difference = close(
            saved["max_selected_vs_latest_rotation_rad"], max(value[1] for value in chosen_errors), max_difference
        )
        reasons[reason] += 1
        armed_count += armed
        retained_count += reason == "event_retained"
        pending, last = next_pending, step
    summary = {
        "queries": 75,
        "armed_events": armed_count,
        "retained_events": retained_count,
        "interventions": closure_count + opening_count,
        "closure_interventions": closure_count,
        "opening_interventions": opening_count,
        "changed_opening_endpoints": changed_count,
        "after_first_intervention_queries": later_count,
        "reasons": dict(sorted(reasons.items())),
    }
    prefix.require(
        produced["first_intervention_step"] == first_change and produced["summary"] == summary,
        "Independent case totals",
    )
    return {
        "case": case,
        "summary": summary,
        "first_intervention_step": first_change,
        "max_numerical_difference": max_difference,
    }


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    item = registration["cases"][index]
    prefix.require(item["case"]["case_id"] == expected_case(index), "Original worker identity")
    refs = item["refs"]
    produced = prefix.read(item["result_ref"])["data"]
    chunks = prefix.read(refs["chunks"])
    tool = prefix.read(refs["tool"])["source_from_tcp"]
    with prefix.checked(refs["steps"]).open() as stream:
        rows = [json.loads(line) for line in stream]
    config = prefix.read(registration["config_ref"])
    data = verify_case(item["case"], produced, chunks, rows, tool, config)
    for ref in refs.values():
        prefix.checked(ref)
    prefix.checked(item["result_ref"])
    print(
        publish(
            OUTPUT / f"case_{index:02d}" / "result.json",
            {
                "complete": True,
                "passed": True,
                "input_refs": refs,
                "producer_case_ref": item["result_ref"],
                "data": data,
            },
        ),
        flush=True,
    )


def execute():
    prefix.environment()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Frozen independent CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "Independent CPU QA source identity")
    inputs, sources = load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True and dry["inputs"] == inputs and dry["source_refs"] == sources,
        "Independent zero-model dry preserves sources and inputs",
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration_ref = publish(OUTPUT / "registration.json", {**inputs, "source_refs": sources})
    specs = [
        {
            "key": f"case_{index:02d}",
            "argv": [
                sys.executable,
                "-m",
                "scripts.verification.reference_ports.check_causal_event_memory",
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
            "Twelve independent worker processes closed and were reaped",
        )
        refs = [prefix.reference(OUTPUT / f"case_{index:02d}" / "result.json") for index in range(12)]
        checked = [prefix.read(ref) for ref in refs]
        prefix.require(
            all(item["complete"] is item["passed"] is True for item in checked)
            and [item["data"]["case"]["case_index"] for item in checked] == list(range(12)),
            "All independent case receipts in original order",
        )
        original = prefix.read(PRODUCER)
        summaries = [item["data"]["summary"] for item in checked]
        reasons = Counter()
        for item in summaries:
            reasons.update(item["reasons"])
        summary = {"cases": 12, "baseline_periods": 7200}
        for key in (
            "queries",
            "armed_events",
            "retained_events",
            "interventions",
            "closure_interventions",
            "opening_interventions",
            "changed_opening_endpoints",
            "after_first_intervention_queries",
        ):
            summary[key] = sum(item[key] for item in summaries)
        summary["reasons"] = dict(sorted(reasons.items()))
        prefix.require(summary == original["summary"], "Independent all-case aggregate")
        for item in inputs["cases"]:
            for ref in item["refs"].values():
                prefix.checked(ref)
            prefix.checked(item["result_ref"])
        prefix.checked(PRODUCER)
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "Unchanged independent source and inputs"
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            case_refs=refs,
            summary=summary,
            max_numerical_difference=max(item["data"]["max_numerical_difference"] for item in checked),
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
