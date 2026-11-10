"""Reconstruct both saved seed42 loaded-gripper histories without physics."""

import argparse
import json
import sys
from pathlib import Path

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.gripper.causal_loaded_history import reconstruct_loaded_history
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_causal_event_prefixes as replay
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/causal_gripper_history_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/causal_gripper_history_20261110.md"
QA = REPO / "outputs/qa/causal_gripper_history_20261110/result.json"
DRY = REPO / "outputs/causal_gripper_history_dry_20261110/result.json"
OUTPUT = REPO / "outputs/causal_gripper_history_20261110"
PREFIX_RESULT = {
    "path": str(REPO / "outputs/causal_event_prefix_replay_20261110/result.json"),
    "sha256": "86a33474a493ce7c5d65cd465cd73efcb731a01411c52cc3644b69f8bc053f65",
}
CASES = [
    {"case_index": 0, "case_id": "task0_init0_s42_raw", "prefix_steps": 136, "task_id": 8},
    {"case_index": 5, "case_id": "task1_init0_s42_terminal058", "prefix_steps": 360, "task_id": 9},
]


def source_refs():
    return replay.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/gripper/causal_loaded_history.py",
            REPO / "src/s2a/research/joint_control/tracking/loaded_velocity_control.py",
            REPO / "src/s2a/research/gripper/gripper_position_hold.py",
            REPO / "src/s2a/research/gripper/coupled_gripper_preview.py",
            REPO / "src/s2a/research/gripper/gripper_reference_preview.py",
            REPO / "src/s2a/research/trajectory_search/velocity_model_planner.py",
            REPO / "src/s2a/research/response_models/causal/velocity_residual_response.py",
            REPO / "tests/research/gripper/test_causal_loaded_history.py",
            REPO / "tests/research/gripper/test_causal_gripper_history_inputs.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == {
            "schema": "s2a.causal-gripper-history.v1",
            "cases": CASES,
            "seed": 42,
            "control_dt_s": 0.05,
            "atol": 1e-10,
            "rtol": 0,
            "max_reconstructed_steps": 496,
            "max_planner_calls": 496,
            "max_candidate_sequences": 3254256,
            "max_workers": 2,
            "worker_timeout_seconds": 1800,
            "physics_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered seed42 gripper-history protocol",
    )
    return ref, config


def read_chunk_prefix(path, count):
    """Decode only the already generated prefix of a top-level chunk array."""
    if type(count) is not int or count < 1:
        raise ValueError("A positive generated-chunk prefix is required")
    text = Path(path).read_text()
    decoder, position, chunks = json.JSONDecoder(), 0, []
    while position < len(text) and text[position].isspace():
        position += 1
    prefix.require(position < len(text) and text[position] == "[", "Generated chunk array")
    position += 1
    for index in range(count):
        while position < len(text) and text[position].isspace():
            position += 1
        if index:
            prefix.require(position < len(text) and text[position] == ",", "Generated chunk separator")
            position += 1
            while position < len(text) and text[position].isspace():
                position += 1
        item, position = decoder.raw_decode(text, position)
        chunks.append(item)
    return chunks


def load_inputs():
    config_ref, config = read_config()
    physical = prefix.read(PREFIX_RESULT)
    prefix.require(
        physical["complete"] is physical["passed"] is physical["sources_unchanged"] is True
        and physical["control_steps"] == 496
        and physical["native_steps"] == 12400
        and physical["newly_selected_commands"] == 0
        and len(physical["case_refs"]) == 2,
        "Both original physical prefixes were reproduced",
    )
    inputs = replay.load_inputs()
    refs = dict(inputs["source_sha256"])
    refs[config_ref["path"]] = config_ref["sha256"]
    refs[PREFIX_RESULT["path"]] = PREFIX_RESULT["sha256"]
    review = prefix.read(prefix.REVIEW, refs)
    source_map = review["source_and_artifact_sha256"]

    def reviewed(path):
        ref = prefix.reference(path)
        prefix.require(source_map.get(ref["path"]) == ref["sha256"], "Reviewed historical gripper input")
        prefix.checked(ref, refs)
        return ref

    cases = []
    for index, (spec, physical_ref) in enumerate(zip(CASES, physical["case_refs"], strict=True)):
        original = inputs["cases"][index]
        item = prefix.read(physical_ref, refs)
        n = spec["prefix_steps"]
        prefix.require(
            original["spec"]["case_index"] == spec["case_index"]
            and original["spec"]["case_id"] == spec["case_id"]
            and original["spec"]["prefix_steps"] == n
            and original["config"]["seed"] == config["seed"]
            and original["config"]["task_id"] == spec["task_id"]
            and item["complete"] is item["passed"] is item["runtime"]["behavioural_replay_verified"] is True
            and item["runtime"]["checked_endpoints"] == item["runtime"]["exact_state_endpoints"] == n + 1,
            "Exact independently verified physical prefix",
        )
        folder = Path(review["cases"][spec["case_index"]]["mechanical"]["case"])
        manifest_ref = reviewed(folder / "manifest.json")
        manifest = prefix.read(manifest_ref, refs)
        controller_ref = reviewed(folder / "controller.json")
        controller = prefix.read(controller_ref, refs)
        steps_ref, chunks_ref = reviewed(folder / "steps.jsonl"), reviewed(folder / "chunks.json")
        tool_ref = reviewed(Path(manifest["runtime_config"]["tool_contract"]))
        parameters_ref = manifest["experiment"]["response_parameters_ref"]
        prefix.checked(parameters_ref, refs)
        prefix.require(
            source_map.get(parameters_ref["path"]) == parameters_ref["sha256"], "Original reviewed response parameters"
        )
        prefix.require(
            manifest["runtime_config"]["seed"] == config["seed"]
            and manifest["experiment"]["control_fps"] == 20
            and manifest["variant"]["governor"] is False
            and manifest["experiment"]["gripper_velocity_envelope"] == controller["gripper_velocity_envelope"]
            and manifest["experiment"]["gripper_position_hold"] == controller["gripper_config"],
            "Original 20 Hz controller, gripper parameters and reference clock",
        )
        cases.append(
            {
                "spec": spec,
                "physical_ref": physical_ref,
                "manifest_ref": manifest_ref,
                "controller_ref": controller_ref,
                "steps_ref": steps_ref,
                "chunks_ref": chunks_ref,
                "tool_ref": tool_ref,
                "parameters_ref": parameters_ref,
            }
        )
    return {"config_ref": config_ref, "cases": cases, "source_sha256": refs}


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        inputs["source_sha256"] == registration["source_sha256"] and source_refs() == registration["source_refs"],
        "Frozen gripper reconstruction sources and inputs",
    )
    item = next(case for case in inputs["cases"] if case["spec"]["case_index"] == index)
    spec, n = item["spec"], item["spec"]["prefix_steps"]
    rows = prefix.read_recorded_prefix(prefix.checked(item["steps_ref"]), n, 600)
    chunks = read_chunk_prefix(prefix.checked(item["chunks_ref"]), n // 8)
    tool = prefix.read(item["tool_ref"])["source_from_tcp"]
    params = prefix.read(item["parameters_ref"])
    controller = prefix.read(item["controller_ref"])
    result = {"complete": False, "passed": False, "case": spec, "automatic_retry": False}
    try:
        _, data = reconstruct_loaded_history(
            rows,
            chunks,
            tool,
            params,
            velocity_envelope=controller["gripper_velocity_envelope"],
            gripper_config=controller["gripper_config"],
        )
        prefix.require(
            data["steps"] == n
            and data["planner_calls"] <= n
            and data["candidate_sequences"] == data["planner_calls"] * 3**8,
            "Actual finite planner costs",
        )
        for key in ("steps_ref", "chunks_ref", "tool_ref", "parameters_ref", "controller_ref", "physical_ref"):
            prefix.checked(item[key])
        result.update(complete=True, passed=True, data=data)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / f"case_{index:02d}" / "result.json", result), flush=True)


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
            "key": f"case_{item['case_index']:02d}",
            "argv": [
                sys.executable,
                "-m",
                "scripts.simulation.runtime.reconstruct_causal_gripper_history",
                "--worker",
                str(item["case_index"]),
            ],
            "log": OUTPUT / f"case_{item['case_index']:02d}" / "worker.log",
        }
        for item in CASES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=1800)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers_ref = publish(OUTPUT / "workers.json", lifecycle)
    result = {
        "complete": False,
        "passed": False,
        "registration_ref": registration_ref,
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
                item["returncode"] == 0 and item["actual_wait"] and item["reaped"] and not item["signal_sent"]
                for item in lifecycle["workers"]
            ),
            "Both registered workers closed and were reaped",
        )
        case_refs = [prefix.reference(OUTPUT / f"case_{item['case_index']:02d}" / "result.json") for item in CASES]
        cases = [prefix.read(ref) for ref in case_refs]
        prefix.require(
            all(item["complete"] is item["passed"] is True for item in cases), "Both complete gripper histories"
        )
        steps = sum(item["data"]["steps"] for item in cases)
        calls = sum(item["data"]["planner_calls"] for item in cases)
        sequences = sum(item["data"]["candidate_sequences"] for item in cases)
        prefix.require(
            steps == 496 and calls <= 496 and sequences == calls * 3**8 and sequences <= 3254256,
            "Registered full-prefix planner budget",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "Unchanged sources and historical inputs"
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            case_refs=case_refs,
            reconstructed_steps=steps,
            planner_calls=calls,
            candidate_sequences=sequences,
            max_recorded_difference=max(item["data"]["max_recorded_difference"] for item in cases),
        )
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", type=int, choices=[0, 5])
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
