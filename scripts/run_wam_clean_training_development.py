"""Own exactly twelve fixed038 development cases and one private model process."""

import argparse
import copy
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
from s2a.clean_training_runtime import GPU, admit_execution, checked, publish, read, reference, require
from s2a.worker_handshake import identity, owned_child

PORT = 31339
CHECKPOINT = "8400_masked"
FIXTURE_SHA = "f522ee333827d51ab66b5b8d1c2a7ebd8ceca3832403b3c54dfa56a821fb2a8e"
CONTROL_REF = dict(
    path=str(REPO / "outputs/clean_training_development_preflight_20261023/dependency_addendum.json"),
    sha256="106f45543f8bc0aa0e336c0ad9b5c01e0bce252a5e5dad0ec3ebbba349e5fda0",
)


def case_order():
    return [
        dict(
            case_index=t * 4 + (seed - 42) * 2 + (variant == "w2"),
            case_id=f"task{t}_init0_s{seed}_{variant}",
            task_index=t,
            record_name=f"task{t}_init0",
            sampling_seed=seed,
            variant=variant,
            expected_requests=75,
        )
        for t in range(3)
        for seed in (42, 43)
        for variant in ("raw", "w2")
    ]


def admit_state(prepared_ref, registration_ref, state_owner_ref, state_verification_ref):
    owner, audit = read(state_owner_ref), read(state_verification_ref)
    common = dict(complete=True, passed=True, prepared_ref=prepared_ref, execution_registration_ref=registration_ref)
    require(
        all(owner.get(k) == audit.get(k) == v for k, v in common.items()), "state owner/audit is failed or unrelated"
    )
    require(
        owner.get("schema_version") == "s2a.clean-training-state-owner.v1"
        and audit.get("schema_version") == "s2a.clean-training-state-verification.v1"
        and owner["verification_ref"] == state_verification_ref
        and owner["producer_ref"] == audit["sampling_ref"],
        "exact state99 owner/producer/audit lineage differs",
    )
    require(
        audit["actual_requests"] == 99
        and audit["actual_denoise_calls"] == audit["actual_head_calls"] == 3465
        and audit["model_initializations"] == 1
        and audit["paired_initial_conditions"] == 48
        and audit["zero_requests_verified"] == 3
        and audit["numerical_contract_passed"] is True
        and audit["eligible_for_development"] is True
        and type(audit["state_gate_passed"]) is bool
        and audit["state_gate"]["passed"] is audit["state_gate_passed"],
        "complete state99 numerical qualification missing",
    )
    require(len(owner["owned_commands"]) == 2, "state owner lacks its two normal owned commands")
    for command in owner["owned_commands"]:
        check_owned_command(command)
    model = owner["owned_commands"][0]
    require(
        all(audit[f"worker_{key}_ref"] == model[key] for key in ("launch", "exit", "release"))
        and audit["gpu_after_ref"] == owner["gpu_after_ref"],
        "state model lifecycle audit differs",
    )
    gpu = read(owner["gpu_after_ref"])
    require(
        gpu["released"] is True and gpu["uuid"] == GPU and gpu["memory_mib"] == 0 and gpu["compute_apps"] == [],
        "state GPU was not normally released",
    )
    return audit


def admit_development(prepared_ref, registration_ref, state_owner_ref, state_verification_ref, *, required_sources):
    prepared, registration, sources = admit_execution(prepared_ref, registration_ref, required_sources=required_sources)
    require(
        registration.get("development_control_ref") == CONTROL_REF
        and registration.get("development") == dict(port=PORT, case_order=case_order()),
        "final registration does not freeze this exact development contract",
    )
    control = read(CONTROL_REF)
    require(
        control["complete"] is control["passed"] is control["control_contract_reuse_eligible"] is True
        and control["new_physics_control_qualification"] is False
        and control["execution_ready"] is False,
        "control preflight is not qualified for conditional reuse",
    )
    for ref in [
        control["cv2_guard_registration"]["helper_ref"],
        control["cv2_guard_registration"]["exact_probe_ref"],
        *control["cv2_guard_registration"]["runtime_source_refs"],
    ]:
        checked(ref)
        require(
            registration["input_sha256"].get(ref["path"], registration["source_sha256"].get(ref["path"]))
            == ref["sha256"],
            "final registration omits the exact qualified cv2 guard binding",
        )
    audit = admit_state(prepared_ref, registration_ref, state_owner_ref, state_verification_ref)
    return prepared, registration, sources, audit, control


def make_case_config(prepared, case):
    base = json.loads((REPO / "configs/wam_short_replan_8_alpha1_20261021.json").read_text())
    require(
        base["fixture_sha256"] == FIXTURE_SHA
        and base["variants"]
        == [
            dict(
                name="position_hold",
                rotation="response",
                gripper="position_hold",
                governor=False,
                model_rotation_alpha=1.0,
            )
        ],
        "qualified raw8 controller differs",
    )
    value = copy.deepcopy(base)
    root = Path(prepared["root"]) / "development/cases" / case["case_id"]
    value.update(
        output_dir=str(root.relative_to(REPO)),
        records=[case["record_name"]],
        sampling_seeds=[case["sampling_seed"]],
        simulation_python=str(REPO / base["simulation_python"]),
        fixture=str(REPO / base["fixture"]),
        notes=["038 fixed development case; legacy GPU label4 is CPU executor metadata only; actual model UUID=" + GPU],
    )
    checkpoint = dict(prepared["model_config"]["checkpoint"], name=CHECKPOINT, gpu=4, port=PORT)
    value["checkpoints"] = [checkpoint]
    return value


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
    prepared, _, sources, audit, _ = admit_development(
        prepared_ref,
        registration_ref,
        state_owner_ref,
        state_verification_ref,
        required_sources=[
            __file__,
            REPO / "scripts/execute_wam_clean_training_case.py",
            REPO / "scripts/verify_wam_clean_training_development.py",
            REPO / "scripts/serve_wam_clean_training.py",
        ],
    )
    common = _flags("prepared", prepared_ref) + _flags("execution-registration", registration_ref)
    common += (
        _flags("state-owner", state_owner_ref) + _flags("state-verification", state_verification_ref) + ["--seed", "42"]
    )
    server = [
        prepared["model_config"]["model_python"],
        "scripts/serve_wam_clean_training.py",
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
            state_gate_passed=audit["state_gate_passed"],
            actual_model_gpu_uuid=GPU,
        )
    root = Path(prepared["root"]) / "development"
    root.mkdir(parents=True, exist_ok=False)
    directory = root / "orchestration"
    directory.mkdir()
    final_directory = Path(prepared["root"]) / "orchestration/DEVELOPMENT"
    final_directory.mkdir(parents=True, exist_ok=False)
    final_owner = dict(
        schema_version="s2a.clean-training-development-audit-owner.v1",
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
        schema_version="s2a.clean-training-development-owner.v1",
        passed=False,
        complete=False,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        state_owner_ref=state_owner_ref,
        state_verification_ref=state_verification_ref,
        w2_fit_ref=audit["w2_fit_ref"],
        w2_verification_ref=audit["w2_verification_ref"],
        readout_ref=audit["readout_ref"],
        cases=[],
        owned_commands=[],
        automatic_retry=False,
        actual_model_gpu_uuid=GPU,
        legacy_executor_gpu_label_only=4,
        state_gate_passed=audit["state_gate_passed"],
        fresh_cases=0,
    )
    deadline = time.monotonic() + prepared["resources"]["development_timeout_s"]
    lock_path = REPO / "outputs/wam_stability_20261021/gpu_locks" / f"{GPU}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    service_dir = directory / "service_process"
    service_dir.mkdir()
    (service_dir / "ownership").mkdir()
    try:
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = resource_observation(GPU)
            result["gpu_before_ref"] = publish(directory / "gpu_before.json", before)
            require(
                before["memory_mib"] == 0 and before["compute_apps"] == [] and before["utilization_gpu"] == 0,
                "private GPU is occupied; no existing process will be touched",
            )
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", PORT))
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
                                config = make_case_config(prepared, case)
                                config_ref = publish(root / "configs" / f"{case['case_id']}.json", config)
                                command = [
                                    str(REPO / "outputs/libero_env311/bin/python"),
                                    "scripts/execute_wam_clean_training_case.py",
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
        str(REPO / ".venv/bin/python"),
        "scripts/verify_wam_clean_training_development.py",
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
            verified["schema_version"] == "s2a.clean-training-development-verification.v1"
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(InterruptedError(f"development signal{signum}")))
    refs = [
        dict(path=str(getattr(args, key).resolve()), sha256=getattr(args, key + "_sha256"))
        for key in ("prepared", "execution_registration", "state_owner", "state_verification")
    ]
    print(json.dumps(execute(*refs, dry_run=args.dry_run)))


if __name__ == "__main__":
    main()
