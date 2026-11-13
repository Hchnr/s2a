"""Run one fixed six-case reset-origin adaptive development matrix."""

import argparse
import sys

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import registered_adaptive_trial as registered
from scripts.simulation.runtime import registered_model_bundle
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_multitask_adaptive_admission as admission
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import live_cv2_admission
from scripts.wam.tracking.preview import run_causal_stream_live_wam as model_owner


def dry_run():
    """Bind all six cases and model bundles without initializing model or physics."""
    prefix.environment()
    config = prefix.read(admission.config_ref())
    admission.validate_config(config)
    return [admission.make_trial(index).dry_run() for index in config["case_indices"]]


def preflight():
    """Check all cases and hash shared model assets once before any GPU work."""
    prefix.environment()
    config = prefix.read(admission.config_ref())
    admission.validate_config(config)
    trials = [admission.make_trial(index) for index in config["case_indices"]]
    inputs = [trial.load_inputs() for trial in trials]
    sources = admission.source_refs()
    before = None
    for trial, data in zip(trials, inputs, strict=True):
        trial.require_config(data)
        checked = trial.check_cpu_admission(data)
        prefix.require(before is None or checked == before, "Every case must share the final CPU source snapshot")
        before = checked
    first = inputs[0]
    prefix.require(
        all(
            data["model_source_hashes"] == first["model_source_hashes"]
            and data["weight_hashes"] == first["weight_hashes"]
            and data["prepared"]["model_config"] == first["prepared"]["model_config"]
            for data in inputs
        ),
        "All cases must use one identical model and weight manifest",
    )
    uuids = list(dict.fromkeys(trial.gpu_uuid for trial in trials))
    devices = {uuid: registered.gpu_observation(uuid) for uuid in uuids}
    prefix.require(
        len(devices) == 4 and all(item["memory_mib"] == 0 and item["compute_apps"] == [] for item in devices.values()),
        "All four registered GPUs must be idle before matrix preflight",
    )
    admission.OUTPUT.mkdir(parents=True, exist_ok=False)
    started = dict(complete=False, config_ref=admission.config_ref(), automatic_retry=False)
    publish(admission.OUTPUT / "preflight_started.json", started)
    try:
        manifests = model_owner.verify_model_manifests(first)
        probe_dir = admission.OUTPUT / "cv2_probe"
        probe_dir.mkdir()
        model_python = first["prepared"]["model_config"]["model_python"]
        probe_ref = publish(probe_dir / "result.json", live_cv2_admission.run_cpu_probe(model_python, probe_dir))
        python_sha = repro.sha256(model_python)
        prefix.require(
            snapshot(prefix.REPO)["sources"] == before and admission.source_refs() == sources,
            "The sources must remain unchanged during the shared preflight",
        )
        case_refs = []
        for trial, data in zip(trials, inputs, strict=True):
            trial.output.mkdir(exist_ok=False)
            case_refs.append(
                publish(
                    trial.output / "preflight.json",
                    dict(
                        complete=True,
                        passed=True,
                        inputs=trial.input_receipt(data),
                        source_refs=sources,
                        qa_ref=prefix.reference(trial.qa),
                        dry_ref=prefix.reference(trial.dry),
                        model_bundle_sha256=registered_model_bundle._digest(trial.model_bundle(data)),
                        gpu_uuid=trial.gpu_uuid,
                        gpu_before=devices[trial.gpu_uuid],
                        cv2_probe_ref=probe_ref,
                        model_python_sha256=python_sha,
                        model_initializations=0,
                        model_requests=0,
                        physics_steps=0,
                        automatic_retry=False,
                        **manifests,
                    ),
                )
            )
        return publish(
            admission.OUTPUT / "preflight.json",
            dict(
                complete=True,
                passed=True,
                config_ref=admission.config_ref(),
                qa_ref=prefix.reference(admission.QA),
                source_refs=sources,
                case_indices=config["case_indices"],
                case_preflight_refs=case_refs,
                gpu_before=devices,
                model_initializations=0,
                model_requests=0,
                physics_steps=0,
                fresh_cases=0,
                automatic_retry=False,
                **manifests,
            ),
        )
    except BaseException as error:
        publish(
            admission.OUTPUT / "preflight_failed.json",
            {**started, "error": {"type": type(error).__name__, "message": str(error)}},
        )
        raise


def run_matrix(phase):
    """Run bounded case owners in two waves, preserving every failure and skip."""
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One matrix phase is required")
    config = prefix.read(admission.config_ref())
    admission.validate_config(config)
    preflight = prefix.read(prefix.reference(admission.OUTPUT / "preflight.json"))
    prefix.require(
        preflight["complete"] is preflight["passed"] is True
        and preflight["config_ref"] == admission.config_ref()
        and preflight["case_indices"] == config["case_indices"]
        and preflight["source_refs"] == admission.source_refs(),
        "The source-matched six-case preflight is required",
    )
    directory = admission.OUTPUT / "orchestration" / phase
    directory.mkdir(parents=True, exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        phase=phase,
        config_ref=admission.config_ref(),
        case_indices=config["case_indices"],
        cases=[],
        waves=[],
        fresh_cases=0,
        automatic_retry=False,
    )
    assigned = set()
    try:
        indices = config["case_indices"]
        for wave_no, start in enumerate(range(0, len(indices), config["max_parallel_cases"])):
            wave = indices[start : start + config["max_parallel_cases"]]
            specs = []
            for index in wave:
                trial = admission.make_trial(index)
                row = dict(case_index=index, gpu_uuid=trial.gpu_uuid, started=False)
                assigned.add(index)
                if phase == "checker":
                    path = trial.output / "producer/result.json"
                    if not path.is_file():
                        row["skip_reason"] = "missing_producer_result"
                    else:
                        producer = prefix.read(prefix.reference(path))
                        if not all(producer.get(key) is True for key in ("complete", "passed", "gpu_released")):
                            row["skip_reason"] = "producer_not_closed"
                else:
                    try:
                        observed = registered.gpu_observation(trial.gpu_uuid)
                        row["gpu_before"] = observed
                        if observed["memory_mib"] != 0 or observed["compute_apps"]:
                            row["skip_reason"] = "gpu_not_idle"
                    except Exception as error:
                        row["skip_reason"] = "gpu_observation_failed"
                        row["error"] = {"type": type(error).__name__, "message": str(error)}
                result["cases"].append(row)
                if "skip_reason" not in row:
                    row["started"] = True
                    specs.append(
                        dict(
                            key=str(index),
                            argv=[
                                sys.executable,
                                "-m",
                                admission.MODULE,
                                "--case-index",
                                str(index),
                                "--phase",
                                phase,
                            ],
                            log=directory / f"case_{index:02d}.log",
                        )
                    )
            if not specs:
                continue
            lifecycle = run_owned_workers(
                specs, cwd=prefix.REPO, timeout_seconds=config["worker_timeout_seconds"] + 600
            )
            for item in lifecycle["workers"]:
                item["log_ref"] = prefix.reference(item.pop("log"))
            result["waves"].append(publish(directory / f"wave_{wave_no:02d}.json", lifecycle))
            by_index = {row["case_index"]: row for row in result["cases"]}
            for item in lifecycle["workers"]:
                index = int(item["key"])
                row = by_index[index]
                row["worker"] = item
                case_result = admission.make_trial(index).output / phase / "result.json"
                if case_result.is_file():
                    row["result_ref"] = prefix.reference(case_result)
                    case = prefix.read(row["result_ref"])
                    row["case_complete"] = case.get("complete") is True
                    row["case_passed"] = case.get("passed") is True
                    if "counts" in case:
                        row["counts"] = case["counts"]
            if lifecycle["launch_error"] is not None:
                result["launch_error"] = lifecycle["launch_error"]
                break
        for index in config["case_indices"]:
            if index not in assigned:
                result["cases"].append(dict(case_index=index, started=False, skip_reason="earlier_wave_interrupted"))
        rows = result["cases"]
        result["complete"] = len(rows) == len(indices) and all(
            row.get("started") is True
            and row.get("worker", {}).get("actual_wait") is True
            and row.get("worker", {}).get("reaped") is True
            and row.get("result_ref") is not None
            for row in rows
        )
        result["passed"] = result["complete"] and all(
            row["worker"]["returncode"] == 0
            and row["worker"]["signal_sent"] is False
            and row["case_complete"] is row["case_passed"] is True
            for row in rows
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        ref = publish(directory / "result.json", result)
        print(ref, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run-matrix", choices=("producer", "checker"))
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", choices=("producer", "checker", "model"))
    parser.add_argument("--case-index", type=int)
    args = parser.parse_args()
    if args.dry_run or args.preflight or args.run_matrix:
        if args.case_index is not None:
            parser.error("--case-index applies only to a case phase or worker")
        if args.dry_run:
            print(dry_run(), flush=True)
        elif args.preflight:
            print(preflight(), flush=True)
        elif not run_matrix(args.run_matrix)["passed"]:
            raise SystemExit(1)
        return
    if args.case_index is None:
        parser.error("--case-index is required for a case phase or worker")
    trial = admission.make_trial(args.case_index)
    if args.phase:
        from scripts.simulation.runtime.reset_multitask_adaptive_execution import execute

        execute(trial, args.phase)
    elif args.worker == "model":
        from scripts.wam.tracking.preview.adaptive_multiseed_model_worker import run_model_worker

        run_model_worker(output=trial.output, gpu_uuid=trial.gpu_uuid, loader=trial.model_worker_inputs)
    else:
        from scripts.simulation.runtime.reset_multitask_adaptive_worker import run_worker

        run_worker(args.case_index, args.worker)


if __name__ == "__main__":
    main()
