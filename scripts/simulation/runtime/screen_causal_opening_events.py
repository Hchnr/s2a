"""Screen all four previously audited first opening-event interventions."""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.preview.opening_event_admission import (
    admit_recorded_state,
    aligned_opening_references,
    recorded_joint_ranges,
)
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import compare_multicase_emergency_retreat as chunks
from scripts.simulation.runtime import replay_causal_event_prefixes as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/causal_opening_event_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/causal_opening_event_feedback_20261111.md"
OUTPUT = REPO / "outputs/causal_opening_event_screen_20261111"
DRY = REPO / "outputs/causal_opening_event_screen_dry_20261111/result.json"
QA = REPO / "outputs/qa/causal_opening_event_screen_20261111/result.json"
CASE_BASE = REPO / "outputs/tracking_preview_wam_development_v4_20261027/development/cases"

CASES = [
    dict(
        case_index=3,
        case_id="task0_init0_s43_terminal058",
        prefix_steps=128,
        source_step=120,
        event_endpoint=134,
        task_id=8,
        target="akita_black_bowl_1",
    ),
    dict(
        case_index=7,
        case_id="task1_init0_s43_terminal058",
        prefix_steps=232,
        source_step=224,
        event_endpoint=236,
        task_id=9,
        target="wine_bottle_1",
    ),
    dict(
        case_index=10,
        case_id="task2_init0_s43_raw",
        prefix_steps=224,
        source_step=216,
        event_endpoint=230,
        task_id=6,
        target="cream_cheese_1",
    ),
    dict(
        case_index=11,
        case_id="task2_init0_s43_terminal058",
        prefix_steps=592,
        source_step=584,
        event_endpoint=596,
        task_id=6,
        target="cream_cheese_1",
    ),
]


def source_refs():
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        Path(prefix.__file__),
        Path(chunks.__file__),
        Path(previous.__file__),
        REPO / "src/s2a/research/joint_control/preview/opening_event_admission.py",
        REPO / "tests/research/joint_control/preview/test_opening_event_admission.py",
        REPO / "tests/research/native_runtime/test_causal_opening_event_screen.py",
    )
    return [prefix.reference(path) for path in paths]


def read_config():
    ref = prefix.reference(CONFIG)
    value = prefix.read(ref)
    prefix.require(
        value
        == {
            "schema": "s2a.causal-opening-event-feedback.v1",
            "cases": CASES,
            "seed": 43,
            "horizon": 8,
            "control_dt_s": 0.05,
            "inner_dt_s": 0.002,
            "joint_margin_rad": 0.05,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "comparison_atol": 1e-10,
            "comparison_rtol": 0,
            "screen_model_initializations": 0,
            "screen_physics_steps": 0,
            "max_workers_per_phase": 8,
            "worker_timeout_seconds": 1800,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered four-case study and original thresholds",
    )
    return ref, value


def current_row(path, n):
    """Decode only the pre-action fields at the intervention endpoint."""
    with Path(path).open() as stream:
        raw = next(itertools.islice(stream, n, n + 1), None)
    prefix.require(raw is not None, "Original current endpoint is present")
    row = json.loads(raw)
    prefix.require(row["step"] == n, "Original current endpoint index")
    return {
        "q": np.asarray(row["qualified_forward_input"]["x_joint_position"], dtype=float).reshape(-1),
        "source_pose": np.asarray(row["measured_source"], dtype=float),
        "opening": float(row["opening_before"]),
    }


def load_context():
    config_ref, config = read_config()
    refs = {config_ref["path"]: config_ref["sha256"]}
    review, shadow, independent = (prefix.read(ref, refs) for ref in (prefix.REVIEW, previous.SHADOW, previous.CHECKER))
    prefix.require(
        review["complete"]
        is review["passed"]
        is shadow["complete"]
        is shadow["passed"]
        is shadow["sources_unchanged"]
        is independent["complete"]
        is independent["passed"]
        is independent["sources_unchanged"]
        is True
        and len(review["cases"]) == len(shadow["case_refs"]) == len(independent["case_refs"]) == 12,
        "Closed original review and independently audited twelve-case shadow",
    )
    source_map = review["source_and_artifact_sha256"]

    def reviewed(path):
        ref = prefix.reference(path)
        prefix.require(source_map.get(ref["path"]) == ref["sha256"], "Reviewed original input identity")
        prefix.checked(ref, refs)
        return ref

    cases = []
    for spec in CASES:
        index, n = spec["case_index"], spec["prefix_steps"]
        original = review["cases"][index]
        case = original["case"]
        prefix.require(
            case["case_id"] == spec["case_id"]
            and case["case_index"] == index
            and case["sampling_seed"] == config["seed"]
            and original["mechanical"]["steps"] == 600
            and original["mechanical"]["chunks"] == 75,
            "Original development case, seed and full budget",
        )
        producer = prefix.read(shadow["case_refs"][index], refs)
        checker = prefix.read(independent["case_refs"][index], refs)
        prefix.require(
            producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True
            and producer["data"]["first_intervention_step"] == n
            and checker["producer_case_ref"] == shadow["case_refs"][index],
            "Bound original first intervention and independent checker",
        )
        case_dir = CASE_BASE / spec["case_id"]
        worker_ref = reviewed(case_dir / "worker_result.json")
        worker = prefix.read(worker_ref, refs)
        folder = Path(worker["physical_case_dir"])
        prefix.require(
            worker["complete"] is worker["passed"] is True
            and worker["physics_steps"] == 600
            and folder.is_relative_to(case_dir)
            and str(folder) == original["mechanical"]["case"],
            "Closed original physical case",
        )
        manifest_ref = reviewed(folder / "manifest.json")
        manifest = prefix.read(manifest_ref, refs)
        prefix.require(
            manifest["runtime_config"]["seed"] == config["seed"]
            and manifest["runtime_config"]["task_id"] == manifest["record"]["task_id"] == spec["task_id"],
            "Original case task and seed",
        )
        item_refs = {
            "worker": worker_ref,
            "manifest": manifest_ref,
            "controller": reviewed(folder / "controller.json"),
            "steps": reviewed(folder / "steps.jsonl"),
            "chunks": reviewed(folder / "chunks.json"),
            "tool": reviewed(Path(manifest["runtime_config"]["tool_contract"])),
            "arrays": reviewed(Path(worker["arrays_ref"]["path"])),
            "endpoints": reviewed(Path(worker["endpoint_receipts_ref"]["path"])),
            "shadow": shadow["case_refs"][index],
            "checker": independent["case_refs"][index],
        }
        prefix.require(item_refs["arrays"] == worker["arrays_ref"], "Original measured endpoint array identity")
        prior = producer["data"]["records"][n // config["horizon"]]
        cases.append({"spec": spec, "refs": item_refs, "shadow": prior, "checked": checker["data"]})
    return {"config_ref": config_ref, "config": config, "cases": cases, "source_sha256": refs}


def screen_case(item, config):
    spec, refs = item["spec"], item["refs"]
    n = spec["prefix_steps"]
    manifest = prefix.read(refs["manifest"])
    current = current_row(prefix.checked(refs["steps"]), n)
    tool = np.asarray(prefix.read(refs["tool"])["source_from_tcp"], dtype=float)
    with np.load(prefix.checked(refs["arrays"]), allow_pickle=False) as data:
        state = np.asarray(data["sim_states"][n], dtype=float).copy()
        source_pose = np.asarray(data["source_pose"][n], dtype=float).copy()
        opening = float(data["opening"][n])
    prefix.require(state.ndim == 1 and len(state) > 8 and np.isfinite(state).all(), "Finite saved current state")
    np.testing.assert_array_equal(state[1:8], current["q"])
    np.testing.assert_allclose(source_pose, current["source_pose"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(opening, current["opening"], atol=1e-12, rtol=0)
    limits = recorded_joint_ranges(manifest["record"]["model_xml"])
    admission = admit_recorded_state(state[1:8], limits, margin_rad=config["joint_margin_rad"])
    chunk_path = prefix.checked(refs["chunks"])
    old = chunks.read_array_item(chunk_path, spec["source_step"] // config["horizon"])
    latest = chunks.read_array_item(chunk_path, n // config["horizon"])
    pair = aligned_opening_references(spec, old, latest, item["shadow"], item["checked"], source_pose @ tool, opening)
    for ref in refs.values():
        prefix.checked(ref)
    return {
        "case": spec,
        "input_refs": refs,
        "current_state_sha256": hashlib.sha256(state.astype(np.float64).tobytes()).hexdigest(),
        "current_opening": opening,
        "admission": admission,
        "reference_sources": {
            "remembered_step": old["step"],
            "remembered_cursor": pair["remembered"]["cursor"],
            "latest_step": latest["step"],
            "latest_cursor": pair["latest"]["cursor"],
            "event_endpoint": spec["event_endpoint"],
        },
        "prior_inner_samples_checked": False,
        "physical_feedback_executed": False,
    }


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    context = load_context()
    prefix.require(
        context["source_sha256"] == registration["source_sha256"] and source_refs() == registration["source_refs"],
        "Frozen screening sources and original inputs",
    )
    item = next(case for case in context["cases"] if case["spec"]["case_index"] == index)
    data = screen_case(item, context["config"])
    print(publish(OUTPUT / f"case_{index:02d}" / "result.json", {"complete": True, "passed": True, "data": data}))


def execute():
    prefix.environment()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Frozen CPU screening QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "QA and screening source identity")
    context, sources = load_context(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["source_sha256"] == context["source_sha256"]
        and dry["source_refs"] == sources,
        "Zero-model dry and screening source identity",
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration = publish(
        OUTPUT / "registration.json", {"source_sha256": context["source_sha256"], "source_refs": sources}
    )
    specs = [
        dict(
            key=f"case_{item['case_index']:02d}",
            argv=[
                sys.executable,
                "-m",
                "scripts.simulation.runtime.screen_causal_opening_events",
                "--worker",
                str(item["case_index"]),
            ],
            log=OUTPUT / f"case_{item['case_index']:02d}" / "worker.log",
        )
        for item in CASES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=context["config"]["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers = publish(OUTPUT / "workers.json", lifecycle)
    result = dict(
        complete=False, passed=False, registration_ref=registration, workers_ref=workers, automatic_retry=False
    )
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == len(CASES)
            and all(
                w["returncode"] == 0 and w["actual_wait"] and w["reaped"] and not w["signal_sent"]
                for w in lifecycle["workers"]
            ),
            "All four screening workers exited and were reaped",
        )
        case_refs = [prefix.reference(OUTPUT / f"case_{spec['case_index']:02d}" / "result.json") for spec in CASES]
        cases = [prefix.read(ref)["data"] for ref in case_refs]
        for path, digest in context["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(source_refs() == sources and snapshot(REPO)["sources"] == before, "Screen sources unchanged")
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            case_refs=case_refs,
            admitted_cases=[
                case["case"]["case_index"] for case in cases if case["admission"]["recorded_endpoint_admitted"]
            ],
            rejected_cases=[
                case["case"]["case_index"] for case in cases if not case["admission"]["recorded_endpoint_admitted"]
            ],
            model_initializations=0,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
        )
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", type=int, choices=[3, 7, 10, 11])
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        context = load_context()
        for case in context["cases"]:
            screen_case(case, context["config"])
        print(
            publish(
                DRY,
                dict(
                    complete=True,
                    passed=True,
                    source_sha256=context["source_sha256"],
                    source_refs=source_refs(),
                    model_initializations=0,
                    physics_steps=0,
                ),
            )
        )
    elif args.run:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
