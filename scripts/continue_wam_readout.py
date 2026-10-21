"""025 explicit continuation after the original resource-observation timeout.

The original failed owner and frozen sources remain unchanged. No feature or
identity forward is available here; only unexecuted CPU/conditional stages run.
"""

import argparse
import datetime
import fcntl
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import run_wam_readout as frozen

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import birth, write_once

REGISTRATION = REPO / "plans/v0.0.1/025_release_observation_addendum_20261021.md"
REGISTRATION_SHA = "ad5b1565517d8c23024b5f26ac21c1c69956558c8e4a652e64d22fe43c209500"
LAUNCH_PROOF = REPO / "outputs/wam_stability_20261021/readout_runtime_observer/features_launch_verified.json"
LAUNCH_PROOF_SHA = "0dbad932ba4ebb649aae966b0c158f3d744e03718aad76c0472ff8c08b1a1818"
OWNER_EXIT = REPO / "outputs/wam_stability_20261021/readout_runtime_observer/features_owner_tool_exit.json"
OWNER_EXIT_SHA = "16393f5e5cb6842eb3bdf86abc933366aac50876029b1633cfa3db8ded211056"
GPU_UUID = "GPU-7f176867-170c-b508-5c89-1cd819ad3b8e"
RELEASE_PEER = REPO / "outputs/wam_stability_20261021/readout_release_review/current_release_verified.json"
RELEASE_PEER_SHA = "c9b076afb246f614d5f4d1a5019964d9972badc5a9c33090b41764485bdff7f8"
INCIDENT_PEER = (
    REPO
    / "outputs/wam_stability_20261021/frozen_wam_readout/review/features_owner_observation_incident_source_peer.json"
)
INCIDENT_PEER_SHA = "2342eb732818a3b8e703e75ab89983a4ee902979aa2afb9fa704a2f6a28a4fd7"


def terminal(expected):
    try:
        observed = birth(expected["pid"], allow_exited=True)
    except (FileNotFoundError, ProcessLookupError):
        return dict(expected=expected, original_birth_terminal=True, current_pid_absent=True)
    done = observed["start_ticks"] != expected["start_ticks"] or observed["proc_state"] in ("Z", "X")
    frozen.require(done, "025 original owned birth is still live")
    return dict(expected=expected, original_birth_terminal=True, observed=observed)


def resource_observation():
    gpu = subprocess.check_output(
        ["nvidia-smi", "--id=6", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True
    ).strip()
    index, uuid, memory = [x.strip() for x in gpu.split(",")]
    frozen.require(index == "6" and uuid == GPU_UUID, "025 current physicalGPU6 UUID differs")
    raw = subprocess.check_output(
        ["nvidia-smi", "--id=6", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    apps = [line for line in raw.splitlines() if len(line.split(",")) >= 2 and line.split(",")[1].strip() == GPU_UUID]
    return dict(
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        physical_gpu=6,
        uuid=uuid,
        memory_mib=int(memory),
        compute_apps=apps,
        raw_compute_query=raw,
    )


def original_evidence(prepared_path, prepared):
    root = Path(prepared["output_root"])
    prior = root / "orchestration/features"
    failed_path = prior / "result.json"
    failed = frozen.read(failed_path)
    frozen.require(
        failed["complete"] is False and failed["passed"] is False and failed["stage"] == "features",
        "025 original failed owner must remain False",
    )
    frozen.require(
        failed["error"]
        == dict(type="TimeoutError", message="025 owned worker exited but GPU6 memory release was not observed"),
        "025 continuation admits only the original resource-observation timeout",
    )
    frozen.require(
        failed["prepared_ref"] == frozen.reference(prepared_path), "025 original owner belongs to another preparation"
    )
    paths = [
        failed_path,
        prior / "model/launch.json",
        prior / "model/exit.json",
        prior / "model/ownership/release.json",
        prior / "model/actual_child_identity.json",
    ]
    launch, exit_record, release, identity = [frozen.read(p) for p in paths[1:]]
    frozen.require(
        exit_record == dict(actual_wait=True, automatic_retry=False, returncode=0),
        "025 original model has no normal actual wait",
    )
    frozen.require(
        release["pid"] == launch["child"]["pid"] and release["start_ticks"] == launch["child"]["start_ticks"],
        "025 original model ownership differs",
    )
    frozen.require(
        release["reaped"] is True
        and release["returncode"] == 0
        and release["signal_sent"] is False
        and release["original_error"] is None
        and release["only_owned_child_signalled"] is True,
        "025 original model did not exit and reap normally without signals",
    )
    frozen.require(
        launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="6", LD_LIBRARY_PATH=""),
        "025 original model environment differs",
    )
    observed = frozen.read(LAUNCH_PROOF, LAUNCH_PROOF_SHA)
    frozen.require(
        observed["passed"] is True
        and observed["launch"] == frozen.reference(paths[1])
        and observed["startup_identity"] == frozen.reference(paths[4]),
        "025 original actual launch proof differs",
    )
    model, owner = observed["actual_model"], observed["actual_owner"]
    frozen.require(
        (model["pid"], model["start_ticks"], owner["pid"], owner["start_ticks"])
        == (862631, 170222073, 862480, 170221627),
        "025 continuation is restricted to the registered original births",
    )
    frozen.require(
        all(model[k] == identity[k] for k in ("pid", "start_ticks", "ppid", "command", "cwd", "environment"))
        and model["ppid"] == owner["pid"],
        "025 actual launch observer is not the original owned model",
    )
    tool_exit = frozen.read(OWNER_EXIT, OWNER_EXIT_SHA)
    frozen.require(
        tool_exit["tool_handle_terminal"] is True
        and tool_exit["tool_reported_exit_code"] == 1
        and tool_exit["exec_session_id"] == 35611
        and tool_exit["original_owner_result"] == frozen.reference(failed_path),
        "025 original owner tool termination proof differs",
    )
    features_path = root / "features/result.json"
    features = frozen.completed(features_path, prepared_path)
    frozen.require(
        features["denoise_calls"] == 249
        and len(features["teacher_calls"]) == 144
        and len(features["zero_identity"]) == 3
        and features["all_original_parameters_frozen"] is True,
        "025 original feature producer did not complete its unique249 budget",
    )
    ledger = features["denoise_ledger"]
    expected = [f"zero:{t}" for t in range(3) for _ in range(35)] + [f"teacher:{i}" for i in range(144)]
    frozen.require(
        len(ledger) == 249
        and all(
            x == dict(index=i, label=label, completed=True)
            for i, (x, label) in enumerate(zip(ledger, expected, strict=True))
        ),
        "025 original actual249 denoise ledger differs",
    )
    references = {str(p.resolve()): sha256(p) for p in paths + [LAUNCH_PROOF, OWNER_EXIT, features_path]}
    release_peer = frozen.read(RELEASE_PEER, RELEASE_PEER_SHA)
    incident_peer = frozen.read(INCIDENT_PEER, INCIDENT_PEER_SHA)
    frozen.require(
        release_peer["passed"] is True
        and release_peer["current_resource_release_verified"] is True
        and release_peer["prepared"] == frozen.reference(prepared_path)
        and release_peer["original_owner_passed"] is False,
        "025 independent current release proof differs",
    )
    frozen.require(
        incident_peer["passed"] is True
        and incident_peer["original_owner_failure_preserved"] is True
        and incident_peer["complete_249_boundary_and144_artifact_roster_verified"] is True
        and incident_peer["zero_full128_exact_verified"] == 3
        and incident_peer["all_teacher_bundle_refs_verified"] == 144,
        "025 independent incident/source roster proof differs",
    )
    for peer_path, peer in ((RELEASE_PEER, release_peer), (INCIDENT_PEER, incident_peer)):
        frozen.check_hashes(peer["input_sha256"])
        references.update(peer["input_sha256"])
        references[str(peer_path)] = sha256(peer_path)
    return dict(
        original_owner_result=frozen.reference(failed_path),
        original_owner_passed=False,
        original_model_actual_wait_exit=0,
        original_owner_tool_exit=tool_exit,
        original_births=[terminal(model), terminal(owner)],
        original_input_sha256=references,
        features_ref=frozen.reference(features_path),
    )


def new_sources(args):
    frozen.require(
        args.registration.resolve() == REGISTRATION
        and args.registration_sha256 == REGISTRATION_SHA
        and sha256(REGISTRATION) == REGISTRATION_SHA,
        "025 resource continuation registration differs",
    )
    paths = [Path(__file__).resolve(), REPO / "tests/test_wam_readout_continuation.py", REGISTRATION]
    return {str(p): sha256(p) for p in paths}


def previous(base, stage, prepared_path, sources):
    result = frozen.completed(base / stage / "result.json", prepared_path)
    frozen.require(
        result["continued_after_resource_observation_timeout"] is True
        and result["continuation_source_sha256"] == sources,
        "025 continuation identity/source differs",
    )
    frozen.check_hashes(result["input_sha256"])
    frozen.require(
        len(result["owned_commands"]) == (1 if stage == "admit" else 2),
        "025 prior continuation has an unexpected command budget",
    )
    for item in result["owned_commands"]:
        for key in ("launch", "exit", "release"):
            frozen.require(
                sha256(item[key]["path"]) == item[key]["sha256"], "025 continuation command evidence changed"
            )
        release = frozen.read(item["release"]["path"])
        launch = frozen.read(item["launch"]["path"])
        frozen.require(
            frozen.read(item["exit"]["path"]) == dict(returncode=0, actual_wait=True, automatic_retry=False),
            "025 previous continuation lacks an actual successful wait",
        )
        frozen.require(
            release["pid"] == launch["child"]["pid"]
            and release["start_ticks"] == launch["child"]["start_ticks"]
            and release["reaped"] is True
            and release["returncode"] == 0
            and release["original_error"] is None,
            "025 previous continuation command did not exit normally",
        )
    return result


def extended_gpu(prepared, peer, command, directory):
    lock_path = REPO / "outputs/wam_stability_20261021/gpu_locks/physical6.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = resource_observation()
        write_once(directory / "gpu_before.json", before)
        frozen.require(
            before["memory_mib"] == 0 and not before["compute_apps"],
            "025 sampling requires currently released physicalGPU6",
        )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 31056))
        frozen.source_guard(prepared, peer)
        entry = frozen.owned_command(
            command,
            frozen.worker_environment(prepared, True),
            directory / "model",
            started=Path(prepared["output_root"]) / "sampling/worker_started.json",
        )
        started = time.monotonic()
        with (directory / "resource_observations.jsonl").open("x") as stream:
            while True:
                item = resource_observation()
                item["seconds_since_actual_owned_wait"] = time.monotonic() - started
                stream.write(frozen.json.dumps(item) + "\n")
                stream.flush()
                if item["memory_mib"] == 0 and not item["compute_apps"]:
                    write_once(directory / "gpu_released.json", item)
                    break
                if time.monotonic() - started >= 120:
                    raise TimeoutError(
                        "025 sampling actual worker exited but resource release was not observed within120seconds"
                    )
                time.sleep(0.1)
        frozen.source_guard(prepared, peer)
        return entry


def execute(args):
    frozen.require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "025 continuation requires S2A cwd and CPU emptyCUDA/LD",
    )
    prepared = frozen.read(args.prepared, args.prepared_sha256)
    peer = frozen.read(args.source_peer, args.source_peer_sha256)
    frozen.require(
        peer["passed"] is True and peer["phase"] == "prepare" and peer["prepared"] == frozen.reference(args.prepared),
        "025 original independent preparation admission differs",
    )
    frozen.require(len(prepared["source_sha256"]) == 53, "025 original53-source preparation changed")
    frozen.source_guard(prepared, peer)
    sources = new_sources(args)
    root = Path(prepared["output_root"])
    base = root / "orchestration/release_continuation"
    features_path = root / "features/result.json"
    proof = original_evidence(args.prepared, prepared)
    if args.stage != "admit":
        previous(base, "admit", args.prepared, sources)
        frozen.admitted_audit(root / "review/features_verified.json", "features", args.prepared, features_path)
    if args.stage == "sampling":
        previous(base, "fit", args.prepared, sources)
        fit_path = root / "fit/fit_result.json"
        fit = frozen.completed(fit_path, args.prepared)
        verification = frozen.admitted_audit(root / "review/fit_verified.json", "fit", args.prepared, fit_path)
        frozen.sampling_admission(fit, verification, args.prepared, features_path, fit_path)
    py = str(REPO / ".venv/bin/python")
    if args.stage == "admit":
        audit = [
            py,
            "scripts/verify_wam_readout.py",
            "--phase",
            "features",
            "--prepared",
            str(args.prepared.absolute()),
            "--features",
            str(features_path),
            "--output",
            str(root / "review/features_verified.json"),
            "--seed",
            "42",
        ]
        worker = None
        producer_result = features_path
    else:
        worker, audit, producer_result = frozen.planned_commands(args, prepared)
    if args.dry_run:
        print(
            frozen.json.dumps(
                dict(
                    stage=args.stage,
                    worker=worker,
                    audit=audit,
                    new_features_calls=0,
                    continued_after_resource_observation_timeout=True,
                )
            )
        )
        return
    directory = base / args.stage
    directory.parent.mkdir(exist_ok=True)
    directory.mkdir()
    review_phase = "features" if args.stage == "admit" else args.stage
    review = root / "review" / f"{review_phase}_verified.json"
    frozen.require(
        not review.exists() and (args.stage == "admit" or not producer_result.exists()),
        "025 continuation refuses existing fit/sampling/audit outputs",
    )
    closure = dict(prepared["source_sha256"], **sources, **proof["original_input_sha256"])
    closure[str(args.prepared.resolve())] = args.prepared_sha256
    closure[str(args.source_peer.resolve())] = args.source_peer_sha256
    if args.stage != "admit":
        for path in (base / "admit/result.json", root / "review/features_verified.json"):
            closure[str(path)] = sha256(path)
    if args.stage == "sampling":
        for path in (base / "fit/result.json", root / "fit/fit_result.json", root / "review/fit_verified.json"):
            closure[str(path)] = sha256(path)
    result = dict(
        schema_version="s2a.frozen-readout-release-continuation.v1",
        stage=args.stage,
        complete=False,
        passed=False,
        prepared_ref=frozen.reference(args.prepared),
        continued_after_resource_observation_timeout=True,
        original_feature_owner_passed=False,
        original_release_evidence=proof,
        continuation_source_sha256=sources,
        input_sha256=closure,
        owned_commands=[],
        new_feature_denoise_calls=0,
        automatic_retry=False,
    )
    try:
        if args.stage == "admit":
            snapshots = directory / "source_snapshot"
            snapshots.mkdir()
            for i, (path, digest) in enumerate(sources.items()):
                target = snapshots / f"{i:02d}_{Path(path).name}"
                shutil.copyfile(path, target)
                frozen.require(sha256(target) == digest, "025 continuation source snapshot differs")
                closure[str(target)] = digest
            lock_path = REPO / "outputs/wam_stability_20261021/gpu_locks/physical6.lock"
            with lock_path.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                resource = resource_observation()
                frozen.require(
                    resource["memory_mib"] == 0 and not resource["compute_apps"],
                    "025 currentGPU6 remains occupied; no continuation admitted",
                )
                resource["original_births"] = proof["original_births"]
                write_once(directory / "current_release_proof.json", resource)
            closure[str(directory / "current_release_proof.json")] = sha256(directory / "current_release_proof.json")
        else:
            if args.stage == "fit":
                entry = frozen.owned_command(worker, frozen.worker_environment(prepared, False), directory / "producer")
            else:
                entry = extended_gpu(prepared, peer, worker, directory)
                result["gpu_released"] = frozen.reference(directory / "gpu_released.json")
                result["resource_observations"] = frozen.reference(directory / "resource_observations.jsonl")
                closure[str(directory / "gpu_released.json")] = sha256(directory / "gpu_released.json")
                closure[str(directory / "resource_observations.jsonl")] = sha256(
                    directory / "resource_observations.jsonl"
                )
            result["owned_commands"].append(entry)
            frozen.completed(producer_result, args.prepared)
        entry = frozen.owned_command(audit, frozen.worker_environment(prepared, False), directory / "independent")
        result["owned_commands"].append(entry)
        checked = frozen.admitted_audit(review, review_phase, args.prepared, producer_result)
        frozen.source_guard(prepared, peer)
        frozen.check_hashes(closure)
        result.update(
            complete=True,
            passed=True,
            producer_result=frozen.reference(producer_result),
            independent_verification=frozen.reference(review),
        )
        if args.stage == "fit":
            result["eligible_for_sampling"] = bool(
                frozen.read(producer_result)["eligible_for_sampling"] and checked["eligible_for_sampling"]
            )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        write_once(directory / "result.json", result)
    print(frozen.json.dumps(dict(stage=args.stage, result=frozen.reference(directory / "result.json"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("admit", "fit", "sampling"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--source-peer", type=Path, required=True)
    parser.add_argument("--source-peer-sha256", required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for sig in (frozen.signal.SIGINT, frozen.signal.SIGTERM):
        frozen.signal.signal(sig, frozen.interrupted)
    execute(args)


if __name__ == "__main__":
    main()
