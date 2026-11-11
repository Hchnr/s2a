"""Own a three-case reset-to-live-WAM pilot and independent physical replay."""

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from s2a.infrastructure.lifecycle.owned_socket_pair import OwnedSocketPairInterrupted, run_owned_socket_pair
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_multitask_live_admission as admission
from scripts.simulation.runtime import reset_multitask_live_audit as audit
from scripts.simulation.runtime import reset_multitask_live_workers as workers
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import live_cv2_admission
from scripts.wam.tracking.preview import run_causal_stream_live_wam as prior_owner

ROOT, OUTPUT, QA, DRY = admission.ROOT, admission.OUTPUT, admission.QA, admission.DRY
MODULE = "scripts.simulation.runtime.run_reset_multitask_live_48"


def gpu_observation(uuid):
    """Query the explicitly selected device and its actual compute processes."""
    prefix.require(uuid in [spec["gpu_uuid"] for spec in admission.CASES], "A registered GPU UUID is required")
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], text=True, timeout=20
    )
    rows = [line.strip().split(", ") for line in query.splitlines()]
    selected = [int(memory) for gpu, memory in rows if gpu == uuid]
    prefix.require(len(selected) == 1, "The registered GPU must be uniquely visible")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
        timeout=20,
    )
    return {
        "memory_mib": selected[0],
        "compute_apps": [line.strip() for line in apps.splitlines() if line.strip().startswith(uuid + ",")],
    }


def idle(observation):
    return observation["memory_mib"] == 0 and observation["compute_apps"] == []


def allocate_ports(indices):
    """Reserve distinct local distributed-service ports during assignment."""
    sockets = []
    try:
        for _index in indices:
            server = socket.socket()
            server.bind(("127.0.0.1", 0))
            sockets.append(server)
        return {str(index): server.getsockname()[1] for index, server in zip(indices, sockets, strict=True)}
    finally:
        for server in sockets:
            server.close()


def dry_run():
    inputs = admission.load_inputs()
    result = dict(
        complete=True,
        passed=True,
        inputs=admission.input_receipt(inputs),
        source_refs=admission.source_refs(),
        model_initializations=0,
        model_requests=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    print(publish(DRY, result), flush=True)


def check_qa_and_dry(inputs):
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"],
        "A complete 32-worker source-matched regression is required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "The QA source snapshot differs")
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == admission.input_receipt(inputs)
        and dry["source_refs"] == admission.source_refs()
        and dry["model_initializations"] == dry["model_requests"] == dry["physics_steps"] == 0,
        "A matching zero-model zero-physics dry is required",
    )
    return before


def preflight():
    prefix.environment()
    inputs = admission.load_inputs()
    before = check_qa_and_dry(inputs)
    prefix.require(not OUTPUT.exists(), "Preflight output must be fresh")
    OUTPUT.mkdir()
    started = {"complete": False, "pid": os.getpid(), "config_ref": inputs["config_ref"]}
    publish(OUTPUT / "preflight_started.json", started)
    try:
        sources = admission.source_refs()
        observations = {spec["gpu_uuid"]: gpu_observation(spec["gpu_uuid"]) for spec in inputs["config"]["cases"]}
        prefix.require(all(idle(value) for value in observations.values()), "Every selected GPU must be idle")
        weights = inputs["weight_hashes"]
        before_stats = prior_owner.file_stats(weights)
        repro.verify_manifest(inputs["model_source_hashes"], workers=32)
        repro.verify_manifest(weights, workers=32)
        after_stats = prior_owner.file_stats(weights)
        prefix.require(before_stats == after_stats, "Checkpoint files changed while being verified")
        probe_dir = OUTPUT / "cv2_probe"
        probe_dir.mkdir()
        probe = live_cv2_admission.run_cpu_probe(inputs["prepared"]["model_config"]["model_python"], probe_dir)
        probe_ref = publish(probe_dir / "result.json", probe)
        prefix.require(
            snapshot(ROOT)["sources"] == before and admission.source_refs() == sources, "Preflight sources changed"
        )
        result = dict(
            complete=True,
            passed=True,
            inputs=admission.input_receipt(inputs),
            source_refs=sources,
            qa_ref=prefix.reference(QA),
            dry_ref=prefix.reference(DRY),
            gpu_before=observations,
            cv2_probe_ref=probe_ref,
            source_files_verified=len(inputs["model_source_hashes"]),
            weight_files_verified=len(weights),
            weight_stats=after_stats,
            model_python_sha256=repro.sha256(inputs["prepared"]["model_config"]["model_python"]),
            model_initializations=0,
            model_requests=0,
            physics_steps=0,
            automatic_retry=False,
        )
        print(publish(OUTPUT / "preflight.json", result), flush=True)
    except BaseException as error:
        publish(
            OUTPUT / "preflight_failed.json",
            {**started, "error": {"type": type(error).__name__, "message": str(error)}},
        )
        raise


def checked_preflight(inputs):
    result = prefix.read(prefix.reference(OUTPUT / "preflight.json"))
    prefix.require(
        result["complete"] is result["passed"] is True
        and result["inputs"] == admission.input_receipt(inputs)
        and result["source_refs"] == admission.source_refs()
        and result["qa_ref"] == prefix.reference(QA)
        and result["dry_ref"] == prefix.reference(DRY)
        and set(result["gpu_before"]) == {item["gpu_uuid"] for item in inputs["config"]["cases"]}
        and all(idle(value) for value in result["gpu_before"].values())
        and result["source_files_verified"] == len(inputs["model_source_hashes"])
        and result["weight_files_verified"] == len(inputs["weight_hashes"]) == 163
        and result["model_python_sha256"] == repro.sha256(inputs["prepared"]["model_config"]["model_python"])
        and result["weight_stats"] == prior_owner.file_stats(inputs["weight_hashes"]),
        "The preflight, complete source identity and checkpoint statistics must match",
    )
    return result


def worker_inputs(index, phase):
    prefix.environment()
    workers.case_directory(phase, index)
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))
    inputs = admission.load_inputs(verify_assets=False)
    checked_preflight(inputs)
    prefix.require(
        registration["phase"] == phase
        and registration["inputs"] == admission.input_receipt(inputs)
        and registration["source_refs"] == admission.source_refs(),
        "Worker inputs differ from the registered phase",
    )
    return inputs, registration


def model_bundle(inputs, index):
    return dict(
        input_receipt=admission.input_receipt(inputs),
        config=inputs["config"],
        prepared=inputs["prepared"],
        bridge=inputs["bridge"],
        template=inputs["template"],
        weight_hashes=inputs["weight_hashes"],
        model_registration=admission.model_registration(
            inputs, index, workers.case_directory("producer", index) / "simulation/runtime"
        ),
    )


def model_worker_inputs(index):
    directory = workers.case_directory("producer", index)
    spec = next(item for item in admission.CASES if item["case_index"] == index)
    prefix.require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == spec["gpu_uuid"]
        and os.environ.get("LD_LIBRARY_PATH") == ""
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and all(os.environ.get(key) == "1" for key in prefix.THREAD_VARS)
        and Path.cwd().resolve() == ROOT,
        "The model worker requires exactly its registered GPU and environment",
    )
    registration = prefix.read(prefix.reference(OUTPUT / "producer/registration.json"))
    bundle = prefix.read(registration["model_bundle_refs"][str(index)])
    preflight = prefix.read(prefix.reference(OUTPUT / "preflight.json"))
    admission.validate_config(bundle["config"])
    expected_weights = {path: detail["sha256"] for path, detail in bundle["prepared"]["weight_files"].items()}
    model_registration = bundle["model_registration"]
    prefix.require(
        registration["phase"] == "producer"
        and registration["inputs"] == bundle["input_receipt"] == preflight["inputs"]
        and registration["source_refs"] == preflight["source_refs"] == admission.source_refs()
        and preflight["complete"] is preflight["passed"] is True
        and set(preflight["gpu_before"]) == {item["gpu_uuid"] for item in admission.CASES}
        and preflight["weight_files_verified"] == 163
        and preflight["qa_ref"] == prefix.reference(QA)
        and preflight["dry_ref"] == prefix.reference(DRY)
        and bundle["weight_hashes"] == expected_weights
        and len(expected_weights) == 163
        and preflight["weight_stats"] == prior_owner.file_stats(expected_weights)
        and preflight["model_python_sha256"] == repro.sha256(bundle["prepared"]["model_config"]["model_python"])
        and model_registration["task"] == spec["task"]
        and model_registration["image_root"] == str((directory / "simulation/runtime").relative_to(ROOT))
        and model_registration["session"] == f"reset_multitask_case{index}_live_s42_20261111"
        and model_registration["initial"]
        == next(item["initial"] for item in bundle["input_receipt"]["cases"] if item["spec"] == spec)
        and model_registration["seed"] == 42
        and model_registration["model_sha256"] == admission.FIXED["model_sha256"]
        and model_registration["max_requests"] == 5
        and model_registration["denoise_steps"] == 35,
        "The model bundle must match its own task, GPU and current endpoint",
    )
    return bundle, preflight


def case_owner(index, phase):
    """Own and reap the actual worker pair or one isolated checker process."""
    directory = workers.case_directory(phase, index)
    spec = next(item for item in admission.CASES if item["case_index"] == index)
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase=phase,
        case_index=index,
        gpu_uuid=spec["gpu_uuid"],
        automatic_retry=False,
        worker_launch_attempted=False,
        model_initializations=None,
        model_requests_attempted=None,
        denoise_calls_attempted=None,
    )
    try:
        inputs, registration = worker_inputs(index, phase)
        simulation_env = physical._cpu_environment(directory / "simulation")
        command = [sys.executable, "-m", MODULE]
        producer = None
        if phase == "producer":
            result["gpu_before"] = gpu_observation(spec["gpu_uuid"])
            prefix.require(idle(result["gpu_before"]), "The case's registered GPU must be idle")
            model_env = prior_owner._model_environment(directory / "model")
            model_env["CUDA_VISIBLE_DEVICES"] = spec["gpu_uuid"]
            model_env["MASTER_PORT"] = str(registration["master_ports"][str(index)])
            specs = {
                "simulation": dict(
                    argv=command + ["--worker", "simulation", "--index", str(index)],
                    env=simulation_env,
                    log=str((directory / "simulation.log").relative_to(ROOT)),
                ),
                "model": dict(
                    argv=[
                        inputs["prepared"]["model_config"]["model_python"],
                        "-m",
                        MODULE,
                        "--worker",
                        "model",
                        "--index",
                        str(index),
                    ],
                    env=model_env,
                    log=str((directory / "model.log").relative_to(ROOT)),
                ),
            }
            result["worker_launch_attempted"] = True
            try:
                lifecycle = run_owned_socket_pair(specs, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
            except OwnedSocketPairInterrupted as error:
                result["workers_ref"] = publish(directory / "workers.json", error.result)
                result["interrupted"] = True
                raise
            result["workers_ref"] = publish(directory / "workers.json", lifecycle)
            prefix.require(lifecycle["passed"] is True, "Both case workers must exit and be reaped")
            model_ref = prefix.reference(directory / "model/result.json")
            model = prefix.read(model_ref)
            result["model_ref"] = model_ref
            result.update(
                model_initializations=model["model_initializations"],
                model_requests_attempted=model["model_requests_attempted"],
                denoise_calls_attempted=model["denoise_calls_attempted"],
            )
            result["gpu_after"] = gpu_observation(spec["gpu_uuid"])
            result["gpu_released"] = idle(result["gpu_after"])
            prefix.require(result["gpu_released"] is True, "The case's registered GPU must be released")
        else:
            pair = registration["producer_cases"][str(index)]
            producer = prefix.read(pair["branch_ref"])
            worker_spec = dict(
                key="checker",
                argv=command + ["--worker", "checker", "--index", str(index)],
                log=directory / "checker.log",
            )
            result["worker_launch_attempted"] = True
            with physical._temporary_environment(simulation_env):
                lifecycle = run_owned_workers(
                    [worker_spec], cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"]
                )
            result["workers_ref"] = publish(directory / "workers.json", lifecycle)
            registered_feedback_phase.require_closed_workers(lifecycle, ["checker"])
            result.update(model_initializations=0, model_requests_attempted=0, denoise_calls_attempted=0)
        branch_ref = prefix.reference(directory / "simulation/result.json")
        branch = prefix.read(branch_ref)
        prefix.require(branch["case"]["case_index"] == index, "The physical branch belongs to a different case")
        counts = audit.summarize_branch(branch, phase, producer=producer)
        if phase == "producer":
            prefix.require(
                model["complete"] is model["passed"] is True
                and model["case_index"] == index
                and model["gpu_uuid"] == spec["gpu_uuid"]
                and model["model_initializations"] == 1
                and model["model_requests_completed"] == model["responses_sent"] == counts["model_requests"]
                and model["denoise_calls_completed"] == 35 * counts["model_requests"],
                "The case's model and physical exchange counts must match",
            )
        else:
            expected_counts = pair["counts"]
            prefix.require(
                counts["control_steps"] == expected_counts["control_steps"]
                and counts["native_steps"] == expected_counts["native_steps"],
                "The checker must replay exactly the actual physical steps",
            )
        result.update(complete=True, passed=True, branch_ref=branch_ref, counts=counts)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if phase == "producer":
            model_path = directory / "model/result.json"
            if model_path.is_file():
                result["available_model_ref"] = prefix.reference(model_path)
                available = json.loads(model_path.read_text())
                result.update(
                    model_initializations=available.get("model_initializations"),
                    model_requests_attempted=available.get("model_requests_attempted"),
                    denoise_calls_attempted=available.get("denoise_calls_attempted"),
                )
            if "gpu_after" not in result:
                try:
                    result["gpu_after"] = gpu_observation(spec["gpu_uuid"])
                    result["gpu_released"] = idle(result["gpu_after"])
                except Exception as error:
                    result["gpu_after_error"] = {"type": type(error).__name__, "message": str(error)}
        branch_path = directory / "simulation/result.json"
        if branch_path.is_file():
            result["available_branch_ref"] = prefix.reference(branch_path)
        print(publish(directory / "result.json", result), flush=True)


def closed_descendants(case_result):
    """Confirm that every launched inner process was actually waited and reaped."""
    ref = case_result.get("workers_ref")
    if ref is None:
        # Only an explicitly rejected pre-launch case has no children to reap.
        return case_result.get("worker_launch_attempted") is False
    lifecycle = prefix.read(ref)
    return all(row.get("actual_wait") is row.get("reaped") is True for row in lifecycle["workers"])


def execute(phase):
    """Run each registered case once; preserve partial failures and fixed denominator."""
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One registered phase is required")
    inputs = admission.load_inputs()
    before = check_qa_and_dry(inputs)
    checked_preflight(inputs)
    sources = admission.source_refs()
    indices = [item["case_index"] for item in inputs["config"]["cases"]]
    producer_cases = {}
    if phase == "checker":
        prior = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(
            prior["owners_closed"]
            is prior["descendants_reaped"]
            is prior["gpu_released"]
            is prior["sources_unchanged"]
            is True,
            "Every producer owner and descendant must be closed and GPUs released before replay",
        )
        for index in indices:
            entry = prior["cases"][str(index)]
            if entry.get("passed") is True:
                producer_cases[str(index)] = dict(
                    branch_ref=entry["branch_ref"], model_ref=entry["model_ref"], counts=entry["counts"]
                )
        prefix.require(producer_cases, "At least one completed producer case is required for replay")
        prefix.require(
            all(idle(gpu_observation(item["gpu_uuid"])) for item in inputs["config"]["cases"]),
            "All selected GPUs must remain released",
        )
    directory = OUTPUT / phase
    directory.mkdir(exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase=phase,
        registered_cases=indices,
        cases={},
        owners_closed=False,
        descendants_reaped=False,
        sources_unchanged=False,
        automatic_retry=False,
        fresh_cases=0,
    )
    try:
        selected_indices = (
            indices if phase == "producer" else [index for index in indices if str(index) in producer_cases]
        )
        for index in selected_indices:
            case_dir = workers.case_directory(phase, index)
            case_dir.mkdir()
            (case_dir / "simulation").mkdir()
            if phase == "producer":
                (case_dir / "model").mkdir()
        model_refs = {}
        ports = {}
        if phase == "producer":
            ports = allocate_ports(indices)
            prefix.require(
                len(set(ports.values())) == len(indices), "Each model must use a distinct local rendezvous port"
            )
            model_refs = {
                str(index): publish(
                    workers.case_directory("producer", index) / "model/registered_inputs.json",
                    model_bundle(inputs, index),
                )
                for index in indices
            }
        registration = publish(
            directory / "registration.json",
            dict(
                phase=phase,
                inputs=admission.input_receipt(inputs),
                source_refs=sources,
                producer_cases=producer_cases,
                model_bundle_refs=model_refs,
                master_ports=ports,
            ),
        )
        result["registration_ref"] = registration
        command = [sys.executable, "-m", MODULE]
        specs = [
            dict(
                key=str(index),
                argv=command + ["--worker", f"case-{phase}", "--index", str(index)],
                log=directory / f"case_{index:02d}.log",
            )
            for index in selected_indices
        ]
        lifecycle = run_owned_workers(specs, cwd=ROOT, timeout_seconds=inputs["config"]["worker_timeout_seconds"] + 30)
        result["workers_ref"] = publish(directory / "workers.json", lifecycle)
        result["owners_closed"] = (
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == len(selected_indices)
            and all(row["actual_wait"] is row["reaped"] is True for row in lifecycle["workers"])
        )
        descendants = []
        for index in indices:
            case_path = workers.case_directory(phase, index) / "result.json"
            if index not in selected_indices:
                result["cases"][str(index)] = {
                    "complete": False,
                    "passed": False,
                    "not_executed": "producer case unavailable",
                }
            elif case_path.is_file():
                ref = prefix.reference(case_path)
                case_result = prefix.read(ref)
                result["cases"][str(index)] = {**case_result, "case_result_ref": ref}
                descendants.append(closed_descendants(case_result))
            else:
                result["cases"][str(index)] = {"complete": False, "passed": False, "error": "case owner result missing"}
                descendants.append(False)
        result["descendants_reaped"] = len(descendants) == len(selected_indices) and all(descendants)
        if phase == "producer":
            result["gpu_after"] = {
                item["gpu_uuid"]: gpu_observation(item["gpu_uuid"]) for item in inputs["config"]["cases"]
            }
            result["gpu_released"] = all(idle(value) for value in result["gpu_after"].values())
        else:
            result["gpu_released"] = True
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        result["sources_unchanged"] = admission.source_refs() == sources and snapshot(ROOT)["sources"] == before
        result["completed_cases"] = sum(entry.get("passed") is True for entry in result["cases"].values())
        result["totals"] = {
            key: sum(entry["counts"][key] for entry in result["cases"].values() if entry.get("passed") is True)
            for key in ("control_steps", "native_steps", "models", "contexts", "model_requests")
        }
        prefix.require(
            result["owners_closed"]
            is result["descendants_reaped"]
            is result["gpu_released"]
            is result["sources_unchanged"]
            is True
            and result["completed_cases"] == len(indices) == 3
            and all(row["returncode"] == 0 and row["signal_sent"] is False for row in lifecycle["workers"])
            and result["totals"]["control_steps"] <= 144
            and result["totals"]["native_steps"] <= 3600
            and result["totals"]["models"] == result["totals"]["contexts"] == 9
            and result["totals"]["model_requests"] <= (15 if phase == "producer" else 0),
            "All registered cases must finish within budget, with all owners and descendants reaped",
        )
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", choices=("case-producer", "case-checker", "simulation", "model", "checker"))
    parser.add_argument("--index", type=int, choices=(0, 4, 8))
    args = parser.parse_args()
    if args.worker is not None and args.index is None:
        parser.error("--index is required for a case worker")
    if args.worker is None and args.index is not None:
        parser.error("--index only applies to a case worker")
    if args.dry_run:
        dry_run()
    elif args.preflight:
        preflight()
    elif args.phase:
        execute(args.phase)
    elif args.worker.startswith("case-"):
        case_owner(args.index, args.worker.removeprefix("case-"))
    elif args.worker == "model":
        workers.model_worker(args.index, model_worker_inputs)
    else:
        workers.simulation_worker(args.index, "producer" if args.worker == "simulation" else "checker", worker_inputs)


if __name__ == "__main__":
    main()
