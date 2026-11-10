"""Replay the registered event-memory selector on saved development references."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from s2a.geometry import pose_error
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.tracking.preview.causal_event_memory import CausalEventMemory, EventMemoryConfig
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import audit_replan_task_events as prior

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/causal_event_memory_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/causal_event_memory_20261110.md"
QA = REPO / "outputs/qa/causal_event_memory_20261110/result.json"
DRY = REPO / "outputs/causal_event_memory_dry_20261110/result.json"
OUTPUT = REPO / "outputs/causal_event_memory_20261110"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            Path(prior.__file__),
            REPO / "src/s2a/research/wam/tracking/preview/causal_event_memory.py",
            REPO / "tests/research/wam/tracking/preview/test_causal_event_memory.py",
            REPO / "tests/research/wam/tracking/preview/test_causal_event_memory_shadow.py",
        )
    ]


def load_inputs():
    original = prior.load_inputs()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    expected = prefix.read(prefix.reference(prior.CONFIG))
    expected["schema"] = "s2a.causal-event-memory.v1"
    prefix.require(config == expected, "Exact registered shadow policy, cases, thresholds and budget")
    return {"config_ref": config_ref, "review_ref": original["review_ref"], "cases": original["cases"]}


def selector_config(config):
    return EventMemoryConfig(
        horizon=config["horizon"],
        stride=config["query_interval"],
        closing_below=config["closing_below"],
        opening_above=config["opening_above"],
        reference_position_m=config["reference_position_scale_m"],
        reference_rotation_rad=config["reference_rotation_scale_rad"],
        current_position_m=config["current_position_scale_m"],
        current_rotation_rad=config["current_rotation_scale_rad"],
    )


def shadow_case(case, chunks, rows, tool, config):
    """Use recorded observations only; later records remain baseline observations."""
    stride, horizon = config["query_interval"], config["horizon"]
    prefix.require(
        len(chunks) == config["queries_per_case"] and len(rows) == config["steps_per_case"] == len(chunks) * stride,
        "Complete registered original timeline",
    )
    tool = np.asarray(tool, dtype=float)
    prefix.require(tool.shape == (4, 4) and np.isfinite(tool).all(), "Finite original TCP transform")
    memory = CausalEventMemory(selector_config(config))
    records, first_change = [], None
    for query, chunk in enumerate(chunks):
        step = query * stride
        prefix.require(chunk["step"] == step, "Original replan clock")
        poses = np.asarray(chunk["effective_targets_tcp"], dtype=float)
        openings = np.asarray(chunk["openings"], dtype=float)
        prefix.require(poses.shape == (horizon, 4, 4) and openings.shape == (horizon,), "Complete current reference")
        for offset in range(stride):
            row = rows[step + offset]
            prefix.require(
                row["step"] == step + offset
                and row["chunk_index"] == query
                and row["chunk_physical_step"] == offset
                and row["opening_target"] == float(openings[offset]),
                "Original executed reference clock and opening",
            )
        measured = np.asarray(rows[step]["measured_source"], dtype=float) @ tool
        measured_opening = float(rows[step]["opening_before"])
        np.testing.assert_allclose(measured, chunk["anchor_tcp"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(measured_opening, chunk["anchor_opening"], atol=1e-12, rtol=0)
        before = memory.state()
        decision = memory.prepare(poses, openings, step=step, measured_tcp=measured, measured_opening=measured_opening)
        source_step, indices = decision["source_step"], decision["source_indices"]
        prefix.require(
            0 <= source_step <= step
            and source_step % stride == 0
            and indices == list(range(step - source_step, step - source_step + stride))
            and indices[-1] < horizon,
            "Selected pose and opening are from the same already generated absolute endpoints",
        )
        source = chunks[source_step // stride]
        np.testing.assert_array_equal(decision["poses"], np.asarray(source["effective_targets_tcp"])[indices])
        np.testing.assert_array_equal(decision["openings"], np.asarray(source["openings"])[indices])
        changed = decision["source"] == "remembered"
        changed_openings = int(np.count_nonzero(decision["openings"] != openings[:stride]))
        prefix.require(not changed or changed_openings > 0, "A remembered event changes the selected opening")
        if changed and first_change is None:
            first_change, label = step, "first_intervention_on_baseline"
        elif first_change is not None:
            label = "baseline_observation_after_intervention"
        else:
            label = "baseline_observation_before_intervention"
        error = pose_error(decision["poses"], poses[:stride])
        memory.commit(decision, completed_steps=stride)
        records.append(
            {
                **decision,
                "query": query,
                "absolute_endpoints": [step + i + 1 for i in range(stride)],
                "baseline_observation_label": label,
                "memory_before": before,
                "memory_after": memory.state(),
                "changed_opening_endpoints": changed_openings,
                "max_selected_vs_latest_position_m": float(np.max(np.linalg.norm(error[:, :3], axis=1))),
                "max_selected_vs_latest_rotation_rad": float(np.max(np.linalg.norm(error[:, 3:], axis=1))),
            }
        )
    return {
        "case": case,
        "queries": len(records),
        "baseline_periods": len(rows),
        "first_intervention_step": first_change,
        "records": records,
        "summary": summarize_records(records),
    }


def summarize_records(records):
    reasons = Counter(record["reason"] for record in records)
    interventions = [record for record in records if record["source"] == "remembered"]
    return {
        "queries": len(records),
        "armed_events": sum(record["armed_new_event"] for record in records),
        "retained_events": reasons["event_retained"],
        "interventions": len(interventions),
        "closure_interventions": sum(record["event"]["direction"] == "closure" for record in interventions),
        "opening_interventions": sum(record["event"]["direction"] == "opening" for record in interventions),
        "changed_opening_endpoints": sum(record["changed_opening_endpoints"] for record in interventions),
        "after_first_intervention_queries": sum(
            record["baseline_observation_label"] == "baseline_observation_after_intervention" for record in records
        ),
        "reasons": dict(sorted(reasons.items())),
    }


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    item = registration["cases"][index]
    prefix.require(item["case"]["case_id"] == prior.expected_case(index), "Original worker identity")
    refs = item["refs"]
    chunks = prefix.read(refs["chunks"])
    tool = prefix.read(refs["tool"])["source_from_tcp"]
    with prefix.checked(refs["steps"]).open() as stream:
        rows = [json.loads(line) for line in stream]
    config = prefix.read(registration["config_ref"])
    data = shadow_case(item["case"], chunks, rows, tool, config)
    for ref in refs.values():
        prefix.checked(ref)
    print(
        publish(
            OUTPUT / f"case_{index:02d}" / "result.json",
            {"complete": True, "passed": True, "input_refs": refs, "data": data},
        ),
        flush=True,
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
                "scripts.wam.tracking.preview.replay_causal_event_memory",
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
        records = [record for case in data for record in case["records"]]
        prefix.require(len(records) == 900, "All registered generated chunks")
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            case_refs=refs,
            summary={
                "cases": 12,
                "baseline_periods": sum(item["baseline_periods"] for item in data),
                **summarize_records(records),
            },
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
