"""Reproduce the two saved seed42 prefixes preceding shadow closure opportunities."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_prefix_replay import map_record_xml, saved_prefix
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/causal_event_prefix_replay_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/causal_event_prefix_replay_20261110.md"
QA = REPO / "outputs/qa/causal_event_prefix_replay_20261110/result.json"
DRY = REPO / "outputs/causal_event_prefix_replay_dry_20261110/result.json"
OUTPUT = REPO / "outputs/causal_event_prefix_replay_20261110"
SHADOW = {
    "path": str(REPO / "outputs/causal_event_memory_20261110/result.json"),
    "sha256": "da654fd4505d2afdb12460b3bc30ce53882a7818b523eb5e85090821cfa8e986",
}
CHECKER = {
    "path": str(REPO / "outputs/causal_event_memory_check_20261110/result.json"),
    "sha256": "eca1389a37bf8021a02d880a2ae5db23db64d777c646c8fed74c02a30d7888ef",
}
GEOMETRY = {
    "path": str(REPO / "outputs/multicase_geometry_roundoff_repair_20261110/result.json"),
    "sha256": "2d51c0c0f2b6649b7759570a8bb1003f74876ba8b0caecd3be983384572fde52",
}
CASES = [
    {
        "case_index": 0,
        "case_id": "task0_init0_s42_raw",
        "prefix_steps": 136,
        "source_step": 128,
        "event_endpoint": 139,
        "task_id": 8,
    },
    {
        "case_index": 5,
        "case_id": "task1_init0_s42_terminal058",
        "prefix_steps": 360,
        "source_step": 344,
        "event_endpoint": 364,
        "task_id": 9,
    },
]


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            Path(prefix.__file__),
            prefix.CORE,
            prefix.RUNTIME,
            prefix.TEST,
            REPO / "tests/research/native_runtime/test_causal_event_prefix_inputs.py",
        )
    ]


def read_config():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        config
        == {
            "schema": "s2a.causal-event-prefix-replay.v1",
            "cases": CASES,
            "seed": 42,
            "atol": 1e-10,
            "rtol": 0,
            "max_control_steps": 496,
            "max_native_steps": 12400,
            "max_models": 6,
            "max_contexts": 6,
            "max_workers": 2,
            "worker_timeout_seconds": 1800,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact two-case seed42 prefix registration and limits",
    )
    return config_ref, config


def validate_shadow_case(spec, shadow, independent, geometry):
    """Bind the already checked first intervention; do not choose replay actions."""
    step, source_step = spec["prefix_steps"], spec["source_step"]
    prefix.require(
        shadow["case"]["case_index"] == independent["case"]["case_index"] == spec["case_index"]
        and shadow["case"]["case_id"] == independent["case"]["case_id"] == spec["case_id"]
        and shadow["first_intervention_step"] == independent["first_intervention_step"] == step,
        "Both shadow assessments identify the registered first intervention",
    )
    record = shadow["records"][step // 8]
    prefix.require(
        record["step"] == step
        and record["source"] == "remembered"
        and record["source_step"] == source_step
        and record["event"]["direction"] == "closure"
        and record["event"]["absolute_endpoint"] == spec["event_endpoint"]
        and record["baseline_observation_label"] == "first_intervention_on_baseline",
        "Registered causally generated closure and original endpoint",
    )
    probe = geometry["queries"][source_step // 8]
    prefix.require(
        geometry["case"]["case_id"] == spec["case_id"]
        and probe["step"] == source_step
        and probe["first_closure_index"] == spec["event_endpoint"] - source_step - 1
        and [v["name"] for v in probe["variants"]] == ["predicted", "zero"]
        and len(probe["variants"][0]["pad_contacts"]) == 2,
        "Original frozen-object geometry and generated event identity",
    )
    return {"shadow": record, "static_geometry": probe}


def load_inputs():
    config_ref, spec = read_config()
    refs = {config_ref["path"]: config_ref["sha256"]}
    review, boot, migrated, shadow, checker, geometry = (
        prefix.read(ref, refs)
        for ref in (prefix.REVIEW, prefix.BOOTSTRAP, prefix.MIGRATED_MANIFEST, SHADOW, CHECKER, GEOMETRY)
    )
    prefix.require(
        review["complete"]
        is review["passed"]
        is shadow["complete"]
        is shadow["passed"]
        is shadow["sources_unchanged"]
        is checker["complete"]
        is checker["passed"]
        is checker["sources_unchanged"]
        is geometry["complete"]
        is geometry["passed"]
        is geometry["sources_unchanged"]
        is True,
        "Original review, independent shadow and geometry evidence closed",
    )
    prefix.require(
        boot["complete"] is boot["passed"] is boot["bootstrap_verified"] is boot["source_unchanged"] is True,
        "Current CPU bootstrap closed",
    )
    shadow_registration = prefix.read(shadow["registration_ref"], refs)
    prefix.require(
        shadow_registration["review_ref"] == prefix.REVIEW
        and len(shadow["case_refs"]) == len(checker["case_refs"]) == 12,
        "Original twelve-case provenance",
    )
    model = boot["bootstrap"]["model_records"][2]
    prefix.require(model["stage"] == "record_xml" and model["completed"] is True, "Previously mapped record XML")
    mapped_ref = {"path": model["xml_ref"]["path"], "sha256": model["xml_ref"]["sha256"]}
    mapped_xml = prefix.checked(mapped_ref, refs).read_bytes()
    assets = {}
    for stage in boot["bootstrap"]["model_records"]:
        prefix.require(stage["completed"] is True, "Every original runtime snapshot stage completed")
        for asset in stage["assets"]:
            raw = asset["reference"]
            ref = {"path": raw["path"], "sha256": raw["sha256"]}
            prefix.checked(ref, refs)
            prefix.require(ref["path"] not in assets or assets[ref["path"]] == ref["sha256"], "Asset identity conflict")
            assets[ref["path"]] = ref["sha256"]
    prefix.require(len(assets) == 96, "Original 96 unique runtime assets")
    source_map = review["source_and_artifact_sha256"]

    def reviewed(path):
        ref = prefix.reference(path)
        prefix.require(source_map.get(ref["path"]) == ref["sha256"], "Original reviewed development input")
        return prefix.checked(ref, refs)

    cases, template_xml = [], None
    for case_spec in CASES:
        index, n = case_spec["case_index"], case_spec["prefix_steps"]
        original = review["cases"][index]
        case = original["case"]
        prefix.require(
            case["case_id"] == case_spec["case_id"]
            and case["sampling_seed"] == spec["seed"]
            and original["mechanical"]["steps"] == 600
            and original["mechanical"]["chunks"] == 75,
            "Original seed42 development case and budget",
        )
        shadow_case = prefix.read(shadow["case_refs"][index], refs)
        checked_case = prefix.read(checker["case_refs"][index], refs)
        geometry_case = prefix.read(geometry["combined_case_refs"][index], refs)
        prefix.require(
            shadow_case["complete"]
            is shadow_case["passed"]
            is checked_case["complete"]
            is checked_case["passed"]
            is True,
            "Closed per-case shadow and independent receipts",
        )
        opportunity = validate_shadow_case(case_spec, shadow_case["data"], checked_case["data"], geometry_case)
        case_dir = (
            REPO / "outputs/tracking_preview_wam_development_v4_20261027/development/cases" / case_spec["case_id"]
        )
        worker = json.loads(reviewed(case_dir / "worker_result.json").read_text())
        prefix.require(
            worker["complete"] is worker["passed"] is True and worker["physics_steps"] == 600, "Original worker closed"
        )
        folder = Path(worker["physical_case_dir"])
        prefix.require(
            folder.is_relative_to(case_dir) and str(folder) == original["mechanical"]["case"],
            "Original physical case folder",
        )
        manifest = json.loads(reviewed(folder / "manifest.json").read_text())
        controller = json.loads(reviewed(folder / "controller.json").read_text())
        rows = prefix.read_recorded_prefix(reviewed(folder / "steps.jsonl"), n, 600)
        receipts = prefix.read_recorded_prefix(reviewed(Path(worker["endpoint_receipts_ref"]["path"])), n + 1, 601)
        arrays_path = reviewed(Path(worker["arrays_ref"]["path"]))
        prefix.require(
            manifest["runtime_config"]["seed"] == spec["seed"]
            and manifest["runtime_config"]["task_id"] == manifest["record"]["task_id"] == case_spec["task_id"],
            "Original task and seed, without substituting another run",
        )
        original_xml = manifest["record"]["model_xml"].encode()
        if template_xml is None:
            template_xml = original_xml
            prefix.require(
                hashlib.sha256(template_xml).hexdigest() == boot["bootstrap"]["original_record_xml_sha256"],
                "Verified original XML template",
            )
        case_xml = map_record_xml(original_xml, template_xml, mapped_xml).decode()
        commands = np.asarray([row["command"] for row in rows], dtype=float)
        with np.load(arrays_path, allow_pickle=False) as archive:
            saved = saved_prefix(archive, n)
        prefix.require(
            commands.shape == (n, 7) and np.isfinite(commands).all(), "Exactly the saved causal command prefix"
        )
        prefix.require(
            hashlib.sha256(np.asarray(saved["sim_states"][0], dtype=np.float64).tobytes()).hexdigest()
            == manifest["record"]["initial_state_sha256"],
            "Original initial state SHA",
        )
        derived = dict(manifest["runtime_config"])
        for key in ("libero_root", "libero_config", "osmesa_library"):
            derived[key] = migrated["runtime_config"][key]
            prefix.require(
                Path(derived[key]).is_absolute() and Path(derived[key]).is_relative_to(REPO), "Migrated runtime path"
            )
        cases.append(
            {
                "spec": case_spec,
                "config": derived,
                "record": manifest["record"],
                "record_xml": case_xml,
                "commands": commands,
                "saved": saved,
                "receipts": receipts,
                "controller": controller,
                "opportunity": opportunity,
            }
        )
    return {
        "spec_ref": config_ref,
        "spec": spec,
        "cases": cases,
        "asset_refs": assets,
        "source_sha256": refs,
        "cv2_admission": boot["cv2_admission"],
    }


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        inputs["source_sha256"] == registration["source_sha256"]
        and inputs["asset_refs"] == registration["asset_refs"]
        and source_refs() == registration["source_refs"],
        "Frozen inputs, assets and sources",
    )
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
    work = OUTPUT / f"case_{index:02d}"
    result = {"complete": False, "passed": False, "case": case["spec"], "automatic_retry": False}
    try:
        guard = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(guard["cv2_probe_ref"], guard["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime

        def progress(receipt):
            completed = receipt.get("control_steps_completed", 0)
            if completed % 32 == 0 or completed == case["spec"]["prefix_steps"]:
                print(
                    json.dumps(
                        {
                            "case_index": index,
                            "checked_endpoints": receipt.get("checked_endpoints", 0),
                            "control_completed": completed,
                            "native_steps": receipt["counters"]["mj_step"],
                        },
                        allow_nan=False,
                    ),
                    flush=True,
                )

        try:
            receipt = run_prefix_runtime(
                case["config"],
                case["record"],
                case["record_xml"],
                case["commands"],
                case["saved"],
                case["receipts"],
                work / "runtime",
                inputs["asset_refs"],
                case["controller"],
                registered_seed=42,
                progress=progress,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        result.update(complete=True, passed=True, runtime=receipt)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(work / "result.json", result), flush=True)


def execute():
    prefix.environment()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Frozen CPU QA and worker lifecycle closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU QA source identity")
    inputs, sources = load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["source_sha256"] == inputs["source_sha256"]
        and dry["asset_refs"] == inputs["asset_refs"]
        and dry["source_refs"] == sources,
        "Zero-model dry preserves frozen inputs and sources",
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration_ref = publish(
        OUTPUT / "registration.json",
        {
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "source_refs": sources,
            "cases": CASES,
            "spec_ref": inputs["spec_ref"],
        },
    )
    specs = [
        {
            "key": f"case_{item['case_index']:02d}",
            "argv": [
                sys.executable,
                "-m",
                "scripts.simulation.runtime.replay_causal_event_prefixes",
                "--worker",
                str(item["case_index"]),
            ],
            "log": OUTPUT / f"case_{item['case_index']:02d}" / "worker.log",
        }
        for item in CASES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=inputs["spec"]["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers_ref = publish(OUTPUT / "workers.json", lifecycle)
    result = {
        "complete": False,
        "passed": False,
        "registration_ref": registration_ref,
        "workers_ref": workers_ref,
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
            all(
                item["complete"] is item["passed"] is True and item["runtime"]["cleanup_complete"] is True
                for item in cases
            ),
            "Both prefix replays and cleanup closed",
        )
        control = sum(item["runtime"]["control_steps_completed"] for item in cases)
        native = sum(item["runtime"]["counters"]["mj_step"]["completed"] for item in cases)
        models = sum(item["runtime"]["counters"]["model_construct"]["completed"] for item in cases)
        contexts = sum(item["runtime"]["counters"]["context"]["completed"] for item in cases)
        prefix.require(
            control == 496 and native == 12400 and models == contexts == 6, "Exact registered actual replay costs"
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "Unchanged replay sources and inputs"
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            case_refs=case_refs,
            control_steps=control,
            native_steps=native,
            model_initializations=models,
            contexts=contexts,
            newly_selected_commands=0,
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
                    "cases": CASES,
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
