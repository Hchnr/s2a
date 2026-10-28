"""Own exactly twelve fixed preview WAM development cases and one private model process."""

import argparse
import fcntl
import json
import os
import signal
import socket
import time
from pathlib import Path

import requests
from run_wam_clean_bridge import check_owned_command, resource_observation
from run_wam_clean_training import environment as training_environment
from run_wam_clean_training import observe_release
from run_wam_readout import owned_command

from s2a.artifacts import REPO
from s2a.clean_training_runtime import GPU, publish, read, reference, require
from s2a.durable_stage_supervisor import accept_detached_launch, launch_detached
from s2a.terminal_development_policy import case_order
from s2a.tracking_preview_wam_control_v5 import make_case_config
from s2a.tracking_preview_wam_execution_v5 import MASTER_PORT, PORT, admit_development
from s2a.worker_handshake import identity, owned_child

_original_training_environment = training_environment


def training_environment(prepared, *, gpu):
    env = _original_training_environment(prepared, gpu=gpu)
    env["MASTER_PORT"] = str(MASTER_PORT)
    return env


def cpu_environment(prepared):
    env = training_environment(prepared, gpu=False)
    env.update(
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        PYTHONPATH=str(REPO / "src") + ":" + str(REPO / "scripts"),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        LP_NUM_THREADS="2",
    )
    return env


def rpc(route, *, body=None, timeout=300):
    require(type(timeout) in (int, float) and 0 < timeout <= 300, "HTTP budget must be at most300 seconds")
    with requests.Session() as session:
        session.trust_env = False
        response = (
            session.get(f"http://127.0.0.1:{PORT}{route}", timeout=timeout)
            if body is None
            else session.post(f"http://127.0.0.1:{PORT}{route}", json=body, timeout=timeout)
        )
        response.raise_for_status()
        return response.json()


def ready(child, birth, prepared_ref, registration_ref, state_verification_ref, deadline):
    while True:
        require(child.poll() is None, "private service exited before readiness")
        require(time.monotonic() < deadline, "fixed development deadline exceeded during initialization")
        try:
            status = rpc("/status", timeout=min(15, max(0.001, deadline - time.monotonic())))
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(0.25)
            continue
        if not status.get("ready"):
            time.sleep(0.25)
            continue
        process = status["process_identity"]
        require(
            process["pid"] == child.pid
            and process["ppid"] == os.getpid()
            and int(process["start_ticks"]) == birth["start_ticks"]
            and process["cuda_visible_devices"] == GPU
            and process["ld_library_path"] == "",
            "private service HTTP/Popen process identity differs",
        )
        observed = identity(child.pid)
        require(
            observed["start_ticks"] == birth["start_ticks"]
            and observed["ppid"] == os.getpid()
            and observed["environment"]["CUDA_VISIBLE_DEVICES"] == GPU
            and observed["environment"]["LD_LIBRARY_PATH"] == "",
            "actual model process environment differs",
        )
        require(
            status["prepared_ref"] == prepared_ref
            and status["execution_registration_ref"] == registration_ref
            and status["state_verification_ref"] == state_verification_ref
            and status["model_initialization_attempts"] == status["model_initializations"] == 1
            and status["actual_requests"] == status["actual_denoise_calls"] == 0,
            "service readiness/budget differs",
        )
        return dict(status=status, actual_process=observed)


def _flags(name, ref):
    return ["--" + name, ref["path"], "--" + name + "-sha256", ref["sha256"]]


def execute(prepared_ref, registration_ref, state_owner_ref, state_verification_ref, *, dry_run=False):
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "development owner requires S2A cwd and explicit empty CUDA/LD",
    )
    prepared, registration, sources, audit, _ = admit_development(
        prepared_ref,
        registration_ref,
        state_owner_ref,
        state_verification_ref,
        required_sources=[
            __file__,
            REPO / "scripts/collect_tracking_preview_wam_development_v5.py",
            REPO / "scripts/verify_tracking_preview_wam_development_v5.py",
            REPO / "scripts/serve_tracking_preview_wam_development_v5.py",
        ],
    )
    common = _flags("prepared", prepared_ref) + _flags("execution-registration", registration_ref)
    common += (
        _flags("state-owner", state_owner_ref) + _flags("state-verification", state_verification_ref) + ["--seed", "42"]
    )
    server = [
        prepared["model_config"]["model_python"],
        "scripts/serve_tracking_preview_wam_development_v5.py",
        *common,
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--gpu-uuid",
        GPU,
    ]
    if dry_run:
        return dict(
            service=server,
            case_order=case_order(),
            model_initializations=0,
            new_physics_steps=0,
            state_gate_passed=audit["development_preconditions_passed"],
            actual_model_gpu_uuid=GPU,
        )
    root = Path(prepared["root"]) / "development"
    root.mkdir(parents=True, exist_ok=False)
    directory = root / "orchestration"
    directory.mkdir()
    final_directory = Path(prepared["root"]) / "orchestration/DEVELOPMENT"
    final_directory.mkdir(parents=True, exist_ok=False)
    final_owner = dict(
        schema_version="s2a.tracking-preview-wam-v5-development-audit-owner.v1",
        passed=False,
        complete=False,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        state_owner_ref=state_owner_ref,
        state_verification_ref=state_verification_ref,
        owned_commands=[],
        automatic_retry=False,
        fresh_performance_authorized=False,
    )
    result = dict(
        schema_version="s2a.tracking-preview-wam-v5-development-owner.v1",
        preview_control_closure_ref=prepared["preview_control_closure_ref"],
        passed=False,
        complete=False,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        state_owner_ref=state_owner_ref,
        state_verification_ref=state_verification_ref,
        candidate_proof_ref=audit["candidate_proof_ref"],
        state_score_ref=audit["state_score_ref"],
        readout_ref=audit["candidate_params_ref"],
        cases=[],
        owned_commands=[],
        automatic_retry=False,
        actual_model_gpu_uuid=GPU,
        state_gate_passed=audit["development_preconditions_passed"],
        fresh_cases=0,
    )
    deadline = time.monotonic() + prepared["resources"]["development_timeout_s"]
    lock_path = REPO / "outputs/wam_stability_20261021/gpu_locks" / f"{GPU}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    service_dir = directory / "service_process"
    service_dir.mkdir()
    (service_dir / "ownership").mkdir()
    try:
        probe = Path(prepared["root"]) / "development_prelaunch_write_probe"
        with probe.open("xb") as stream:
            stream.write(b"061 explicit development write probe\n")
            stream.flush()
            os.fsync(stream.fileno())
        probe.unlink()
        space = os.statvfs(prepared["root"])
        available = space.f_bavail * space.f_frsize
        require(available >= 10 * 1024**3, "061 insufficient available space")
        result["prelaunch_write_probe"] = dict(passed=True, available_bytes=available)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = resource_observation(GPU)
            result["gpu_before_ref"] = publish(directory / "gpu_before.json", before)
            require(
                before["memory_mib"] == 0 and before["compute_apps"] == [] and before["utilization_gpu"] == 0,
                "private GPU is occupied; no existing process will be touched",
            )
            for registered_port in (PORT, MASTER_PORT):
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", registered_port))
            with (service_dir / "command.log").open("x") as stream:
                try:
                    with owned_child(
                        server,
                        environment=training_environment(prepared, gpu=True),
                        cwd=REPO,
                        folder=service_dir / "ownership",
                        stream=stream,
                    ) as (child, birth):
                        launch = publish(
                            service_dir / "launch.json",
                            dict(
                                child=birth,
                                command=server,
                                selected_environment=dict(CUDA_VISIBLE_DEVICES=GPU, LD_LIBRARY_PATH=""),
                                automatic_retry=False,
                            ),
                        )
                        result["service_launch_ref"] = launch
                        result["readiness_ref"] = publish(
                            directory / "service_readiness.json",
                            ready(child, birth, prepared_ref, registration_ref, state_verification_ref, deadline),
                        )
                        active = None
                        try:
                            for case in case_order():
                                require(
                                    child.poll() is None and time.monotonic() < deadline,
                                    "service/development deadline failed",
                                )
                                bind = rpc(
                                    "/bind_case", body=case, timeout=min(300, max(0.001, deadline - time.monotonic()))
                                )
                                active = case["case_id"]
                                config = make_case_config(
                                    prepared,
                                    registration,
                                    prepared_ref=prepared_ref,
                                    registration_ref=registration_ref,
                                    case=case,
                                )
                                config_ref = publish(root / "configs" / f"{case['case_id']}.json", config)
                                command = [
                                    str(REPO / "outputs/libero_env311/bin/python"),
                                    "scripts/collect_tracking_preview_wam_development_v5.py",
                                    *common,
                                    *_flags("case-config", config_ref),
                                    "--case-index",
                                    str(case["case_index"]),
                                ]
                                entry = owned_command(
                                    command,
                                    cpu_environment(prepared),
                                    directory / case["case_id"],
                                    timeout=max(0.001, deadline - time.monotonic()),
                                )
                                result["owned_commands"].append(entry)
                                check_owned_command(entry)
                                status = rpc("/status", timeout=min(300, max(0.001, deadline - time.monotonic())))
                                require(
                                    status["case_index"] == case["case_index"]
                                    and status["current_case_id"] == case["case_id"]
                                    and status["case_requests"] == 75
                                    and status["actual_requests"] == (case["case_index"] + 1) * 75,
                                    "private service case/request budget differs",
                                )
                                worker = Path(config["output_dir"]) / "worker_result.json"
                                worker_ref = reference(worker)
                                value = read(worker_ref)
                                require(
                                    value["complete"] is value["passed"] is True and value["case"] == case,
                                    "physics worker did not complete the exact registered case",
                                )
                                result["cases"].append(
                                    dict(
                                        case=case,
                                        config_ref=config_ref,
                                        worker_ref=worker_ref,
                                        owned_command=entry,
                                        bind_response=bind,
                                        status_after=status,
                                    )
                                )
                            rpc(
                                "/finish",
                                body=dict(completed_cases=12, completed_requests=900),
                                timeout=min(300, max(0.001, deadline - time.monotonic())),
                            )
                            code = child.wait(timeout=max(0.001, deadline - time.monotonic()))
                            result["service_exit_ref"] = publish(
                                service_dir / "exit.json",
                                dict(returncode=code, actual_wait=True, automatic_retry=False),
                            )
                            require(code == 0, "private service failed after complete900 requests")
                        except BaseException:
                            if child.poll() is None:
                                try:
                                    rpc(
                                        "/abort",
                                        body=dict(
                                            case_id=active, reason="development owner stopped after a retained failure"
                                        ),
                                        timeout=120,
                                    )
                                    code = child.wait(timeout=max(0.001, min(120, deadline - time.monotonic())))
                                    if not (service_dir / "exit.json").exists():
                                        result["service_exit_ref"] = publish(
                                            service_dir / "exit.json",
                                            dict(returncode=code, actual_wait=True, automatic_retry=False),
                                        )
                                except Exception as abort_error:
                                    result["abort_error"] = dict(
                                        type=type(abort_error).__name__, message=str(abort_error)
                                    )
                            raise
                finally:
                    release = service_dir / "ownership/release.json"
                    if release.exists():
                        result["service_release_ref"] = reference(release)
                        result["gpu_after_ref"] = observe_release(directory)
            service_command = dict(
                launch=result["service_launch_ref"],
                exit=result["service_exit_ref"],
                release=result["service_release_ref"],
            )
            check_owned_command(service_command)
            result["service_owned_command"] = service_command
        result["service_ref"] = reference(root / "service/result.json")
        service = read(result["service_ref"])
        require(
            service["passed"] is service["complete"] is True
            and service["actual_requests"] == 900
            and service["actual_denoise_calls"] == service["actual_head_calls"] == 31500
            and service["model_initializations"] == 1,
            "private complete generation budget differs",
        )
        for path, digest in sources.items():
            require(reference(path)["sha256"] == digest, "development source changed")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        final_owner["error"] = result["error"]
        publish(final_directory / "result.json", final_owner)
        raise
    finally:
        publish(directory / "result.json", result)
    final_owner["development_ref"] = reference(directory / "result.json")
    final_owner["service_owned_command"] = result["service_owned_command"]
    final_owner["gpu_after_ref"] = result["gpu_after_ref"]
    output = Path(prepared["root"]) / "review/DEVELOPMENT_verified.json"
    command = [
        str(REPO / "outputs/libero_env311/bin/python"),
        "scripts/verify_tracking_preview_wam_development_v5.py",
        *_flags("prepared", prepared_ref),
        *_flags("execution-registration", registration_ref),
        *_flags("development", final_owner["development_ref"]),
        "--output",
        str(output),
        "--seed",
        "42",
    ]
    try:
        require(time.monotonic() < deadline, "fixed development deadline exhausted before independent audit")
        entry = owned_command(
            command,
            cpu_environment(prepared),
            final_directory / "independent",
            timeout=min(prepared["resources"]["audit_timeout_s"], deadline - time.monotonic()),
        )
        final_owner["owned_commands"].append(entry)
        check_owned_command(entry)
        final_owner["verification_ref"] = reference(output)
        verified = read(final_owner["verification_ref"])
        require(
            verified["schema_version"] == "s2a.tracking-preview-wam-v5-development-verification.v1"
            and verified["passed"] is verified["complete"] is True
            and verified["prepared_ref"] == prepared_ref
            and verified["execution_registration_ref"] == registration_ref
            and verified["development_ref"] == final_owner["development_ref"],
            "exact independent development audit failed",
        )
        final_owner.update(complete=True, passed=True, score=verified["score"])
    except BaseException as error:
        final_owner["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        publish(final_directory / "result.json", final_owner)
    return reference(final_directory / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "execution-registration", "state-owner", "state-verification"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(InterruptedError(f"development signal{signum}")))
    refs = [
        dict(path=str(getattr(args, key).resolve()), sha256=getattr(args, key + "_sha256"))
        for key in ("prepared", "execution_registration", "state_owner", "state_verification")
    ]
    if args.dry_run:
        print(json.dumps(execute(*refs, dry_run=True)))
        return
    prepared, _, sources, _, _ = admit_development(*refs, required_sources=[__file__])
    command = [str(REPO / ".venv/bin/python"), "scripts/run_tracking_preview_wam_development_v5.py"]
    for name, ref in zip(
        ("prepared", "execution-registration", "state-owner", "state-verification"), refs, strict=True
    ):
        command += _flags(name, ref)
    command += ["--seed", "42"]
    folder = Path(prepared["root"]) / "supervisor"
    if args.launch:
        print(json.dumps(launch_detached(command, directory=folder, source_sha256=sources)))
        return
    accept_detached_launch(folder, command=command, source_sha256=sources)
    print(json.dumps(execute(*refs, dry_run=False)))


if __name__ == "__main__":
    main()
