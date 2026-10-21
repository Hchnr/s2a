"""025 explicit owned stages: GPU6 features, CPU fit, gated GPU6 sampling."""

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import identity, owned_child, write_once

EXTERNAL_HEAD = "8ed68a9722a14e8f6c21fa0c6c700e1617b7b73e"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    return dict(path=str(Path(path).absolute()), sha256=sha256(path))


def read(path, digest=None):
    if digest is not None:
        require(sha256(path) == digest, f"025 evidence SHA changed: {path}")
    return json.loads(Path(path).read_text())


def check_hashes(mapping):
    for path, digest in mapping.items():
        require(sha256(path) == digest, f"025 frozen source changed: {path}")


def external_identity(repo):
    head = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"], text=True)
    require(
        head == EXTERNAL_HEAD and not dirty.strip(),
        "025 external tracked source must remain the frozen clean8ed commit",
    )
    return dict(head=head, tracked_source_clean=True)


def source_guard(prepared, peer):
    for mapping in (
        prepared["input_sha256"],
        prepared["source_sha256"],
        prepared["snapshot_sha256"],
        peer["input_sha256"],
    ):
        check_hashes(mapping)
    require(
        str(Path(__file__).resolve()) in prepared["source_sha256"], "025 preparation omitted its actual owner source"
    )
    require(len(prepared["weight_files"]) == 163, "025 actual8400 roster must contain163 files")
    for path, item in prepared["weight_files"].items():
        stat = Path(path).stat()
        require(
            (stat.st_size, stat.st_mtime_ns) == (item["bytes"], item["mtime_ns"]), "025 frozen8400 weight stat differs"
        )
    actual = external_identity(prepared["config"]["cosmos_repo"])
    require(actual == prepared["external_cosmos"], "025 external source differs from prepared admission")
    return actual


def completed(path, prepared_path):
    result = read(path)
    require(
        result.get("complete") is True and result.get("passed") is True and "error" not in result,
        f"025 prior stage incomplete/rejected: {path}",
    )
    require(result["prepared_ref"] == reference(prepared_path), "025 prior result belongs to another preparation")
    return result


def admitted_audit(path, phase, prepared_path, result_path):
    result = read(path)
    require(
        result.get("passed") is True and result.get("phase") == phase and "error" not in result,
        f"025 independent {phase} audit did not pass",
    )
    require(result["prepared"] == reference(prepared_path), "025 independent audit is bound to another preparation")
    check_hashes(result["input_sha256"])
    require(
        result["input_sha256"].get(str(Path(result_path).resolve())) == sha256(result_path),
        "025 independent audit did not verify this exact result",
    )
    return result


def owner_completed(root, stage, prepared_path):
    directory = root / "orchestration" / stage
    owner = completed(directory / "result.json", prepared_path)
    require(owner["stage"] == stage, "025 owner stage identity differs")
    require(len(owner["owned_commands"]) == 2, "025 preceding stage must own its producer and independent audit")
    for item in owner["owned_commands"]:
        launch = read(item["launch"]["path"], item["launch"]["sha256"])
        exit_record = read(item["exit"]["path"], item["exit"]["sha256"])
        release = read(item["release"]["path"], item["release"]["sha256"])
        require(
            exit_record == dict(returncode=0, actual_wait=True, automatic_retry=False),
            "025 preceding command lacks a successful actual wait",
        )
        require(
            release["pid"] == launch["child"]["pid"] and release["start_ticks"] == launch["child"]["start_ticks"],
            "025 preceding owned birth changed",
        )
        require(
            release["reaped"] is True and release["returncode"] == 0 and release["original_error"] is None,
            "025 preceding owned command was not normally reaped",
        )
        require(release["only_owned_child_signalled"] is True, "025 preceding command ownership is not certified")
    if stage != "fit":
        saved = owner["gpu_release"]
        gpu = read(saved["path"], saved["sha256"])
        require(
            gpu["released"] is True and gpu["memory_mib"] == 0 and gpu["physical_gpu"] == 6,
            "025 preceding model did not release GPU6",
        )
    return owner


def sampling_admission(fit, verification, prepared_path, features_path, fit_path):
    require(
        fit.get("complete") is True and fit.get("passed") is True and "error" not in fit,
        "025 fit producer did not complete",
    )
    require(
        fit["prepared_ref"] == reference(prepared_path) and fit["features_ref"] == reference(features_path),
        "025 fit source identity differs",
    )
    require(fit["producer_fit_groups"] == fit["fit_attempts"] == 2, "025 fit did not complete its fixed two groups")
    require(
        fit.get("eligible_for_sampling") is True and fit["teacher_gate"]["passed"] is True,
        "025 producer teacher gate rejected sampling",
    )
    require(
        verification.get("phase") == "fit"
        and verification.get("passed") is True
        and verification.get("fit_verified") is True,
        "025 independent fit reconstruction did not pass",
    )
    require(
        verification.get("eligible_for_sampling") is True and verification["teacher_gate"]["passed"] is True,
        "025 independent teacher gate rejected sampling",
    )
    require(verification["prepared"] == reference(prepared_path), "025 independent fit prepared identity differs")
    require(
        verification["input_sha256"].get(str(Path(fit_path).resolve())) == sha256(fit_path),
        "025 independent fit did not bind this producer result",
    )
    for model in fit["models"].values():
        require(sha256(model["path"]) == model["sha256"], "025 fitted model bytes changed")


def worker_environment(prepared, gpu):
    c = prepared["config"]
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="6" if gpu else "",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=c["cosmos_repo"] + ":" + str(REPO / "src"),
        MASTER_PORT="31056",
        COSMOS_TRAINING="0",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        TORCHINDUCTOR_CACHE_DIR=str(Path(prepared["output_root"]) / "model_cache"),
    )


def observe_started(child, birth, command, started, directory, *, expected_cuda="6", timeout=30.0):
    """Observe exec only after the actual Python worker publishes its receipt."""
    deadline, observations = time.monotonic() + timeout, []
    require(expected_cuda in ("", "6"), "025 admits only CPU or physicalGPU6 workers")
    while True:
        if child.poll() is not None:
            raise ChildProcessError("025 worker exited before startup identity admission")
        if started.exists():
            try:
                receipt = read(started)
            except (FileNotFoundError, json.JSONDecodeError):
                # A source-frozen worker may publish via open/write/close. An
                # incomplete observation never grants admission or restarts it.
                observations.append(dict(startup_receipt_incomplete=True))
                if time.monotonic() >= deadline:
                    raise TimeoutError("025 startup receipt remained incomplete") from None
                time.sleep(0.01)
                continue
            require(
                receipt["pid"] == child.pid and receipt["ppid"] == os.getpid(),
                "025 startup receipt is not this owned child",
            )
            require(
                receipt["cuda"] == expected_cuda and receipt["ld"] == "",
                "025 actual worker startup environment differs",
            )
            observed = identity(child.pid)
            observations.append(observed)
            require(
                observed["start_ticks"] == birth["start_ticks"] and observed["ppid"] == os.getpid(),
                "025 model worker birth/parent changed",
            )
            require(
                observed["command"] == command and observed["cwd"] == str(REPO),
                "025 actual model command/cwd differs after startup",
            )
            expected = dict(CUDA_VISIBLE_DEVICES=expected_cuda, LD_LIBRARY_PATH="")
            for key, value in expected.items():
                require(
                    observed["environment"][key] in (None, value),
                    "025 actual model environment contains a forbidden value",
                )
            if all(observed["environment"][k] is not None for k in expected):
                write_once(directory / "identity_observations.json", observations)
                write_once(directory / "actual_child_identity.json", observed)
                return observed
        if time.monotonic() >= deadline:
            raise TimeoutError("025 actual worker exec identity remained unknown")
        time.sleep(0.01)


def owned_command(command, environment, directory, *, started=None, timeout=7200):
    """Retain the exact Popen handle through actual wait and cleanup."""
    directory.mkdir()
    ownership = directory / "ownership"
    ownership.mkdir()
    with (
        (directory / "command.log").open("x") as stream,
        owned_child(command, environment=environment, cwd=REPO, folder=ownership, stream=stream) as (child, birth),
    ):
        write_once(
            directory / "launch.json",
            dict(
                child=birth,
                command=command,
                selected_environment={k: environment[k] for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")},
                automatic_retry=False,
            ),
        )
        if started is not None:
            observe_started(
                child, birth, command, started, directory, expected_cuda=environment["CUDA_VISIBLE_DEVICES"]
            )
        code = child.wait(timeout=timeout)
        write_once(directory / "exit.json", dict(returncode=code, actual_wait=True, automatic_retry=False))
        if code:
            raise ChildProcessError(f"025 command failed with{code}; no automatic retry")
    return dict(
        launch=reference(directory / "launch.json"),
        exit=reference(directory / "exit.json"),
        release=reference(ownership / "release.json"),
    )


def gpu_memory():
    return int(
        subprocess.check_output(
            ["nvidia-smi", "--id=6", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        ).strip()
    )


def run_gpu(prepared, peer, command, stage, directory):
    lock_path = REPO / "outputs/wam_stability_20261021/gpu_locks/physical6.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = gpu_memory()
        write_once(directory / "gpu_before.json", dict(physical_gpu=6, memory_mib=before))
        require(before == 0, "025 requires GPU6 memory0; existing processes remain untouched")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 31056))
        source_guard(prepared, peer)
        entry = owned_command(
            command,
            worker_environment(prepared, True),
            directory / "model",
            started=Path(prepared["output_root"]) / stage / "worker_started.json",
        )
        deadline, observations = time.monotonic() + 10, []
        while True:
            memory = gpu_memory()
            observations.append(memory)
            if memory == 0:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("025 owned worker exited but GPU6 memory release was not observed")
            time.sleep(0.05)
        write_once(
            directory / "gpu_after.json",
            dict(physical_gpu=6, memory_mib=memory, memory_observations_mib=observations, released=True),
        )
        source_guard(prepared, peer)
        return entry


def planned_commands(args, prepared):
    root = Path(prepared["output_root"])
    py = str(REPO / ".venv/bin/python")
    model_py = str(Path(prepared["config"]["cosmos_repo"]) / ".venv/bin/python")
    common = ["--prepared", str(args.prepared.absolute()), "--prepared-sha256", args.prepared_sha256]
    features, fit, verified_fit = (
        root / "features/result.json",
        root / "fit/fit_result.json",
        root / "review/fit_verified.json",
    )
    if args.stage == "features":
        worker = [model_py, "scripts/run_wam_readout_features.py", *common, "--seed", "42"]
        result = features
    elif args.stage == "fit":
        worker = [
            py,
            "scripts/fit_wam_readout.py",
            *common,
            "--features",
            str(features),
            "--features-sha256",
            sha256(features),
            "--seed",
            "42",
        ]
        result = fit
    else:
        worker = [
            model_py,
            "scripts/run_wam_readout_sampling.py",
            *common,
            "--fit",
            str(fit),
            "--fit-sha256",
            sha256(fit),
            "--fit-verification",
            str(verified_fit),
            "--fit-verification-sha256",
            sha256(verified_fit),
            "--seed",
            "42",
        ]
        result = root / "sampling/result.json"
    audit = [
        py,
        "scripts/verify_wam_readout.py",
        "--phase",
        args.stage,
        "--prepared",
        str(args.prepared.absolute()),
        "--features",
        str(features),
    ]
    if args.stage in ("fit", "sampling"):
        audit += ["--fit", str(fit)]
    if args.stage == "sampling":
        audit += ["--fit-verification", str(verified_fit), "--sampling", str(result)]
    audit += ["--output", str(root / "review" / f"{args.stage}_verified.json"), "--seed", "42"]
    return worker, audit, result


def execute(args):
    require(Path.cwd().resolve() == REPO, "025 owner must run from S2A")
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "025 owner requires explicitly empty CUDA/LD",
    )
    prepared = read(args.prepared, args.prepared_sha256)
    peer = read(args.source_peer, args.source_peer_sha256)
    require(
        peer.get("passed") is True
        and peer.get("phase") == "prepare"
        and peer.get("preparation_verified") is True
        and peer.get("prepared") == reference(args.prepared),
        "025 prepare-only independent source review did not admit this preparation",
    )
    config, root = prepared["config"], Path(prepared["output_root"])
    require(
        root == (REPO / config["output_dir"]).resolve() and root.is_relative_to(REPO / "outputs"),
        "025 owner output root differs",
    )
    require(
        config["checkpoint"]["gpu"] == 6 and config["hidden_size"] == 1280 and config["ridge"] == 0.1,
        "025 registered GPU/readout changed",
    )
    require(
        config["budget"]
        == dict(
            zero_identity_generations=3,
            teacher_denoise_calls=144,
            cpu_fits=2,
            validation_generations=96,
            total_denoise_calls=3609,
            physics_steps=0,
            fresh_cases=0,
        ),
        "025 registered stage budgets changed",
    )
    before = source_guard(prepared, peer)
    if args.stage in ("fit", "sampling"):
        owner_completed(root, "features", args.prepared)
        completed(root / "features/result.json", args.prepared)
        admitted_audit(root / "review/features_verified.json", "features", args.prepared, root / "features/result.json")
    if args.stage == "sampling":
        owner_completed(root, "fit", args.prepared)
        fit_path, features_path = root / "fit/fit_result.json", root / "features/result.json"
        fit = completed(fit_path, args.prepared)
        verified = admitted_audit(root / "review/fit_verified.json", "fit", args.prepared, fit_path)
        sampling_admission(fit, verified, args.prepared, features_path, fit_path)
    worker, audit, result_path = planned_commands(args, prepared)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    stage=args.stage,
                    worker=worker,
                    audit=audit,
                    physical_gpu=6 if args.stage != "fit" else None,
                    new_model_calls=0,
                    new_fit_groups=0,
                )
            )
        )
        return
    directory = root / "orchestration" / args.stage
    directory.parent.mkdir(exist_ok=True)
    directory.mkdir()
    review_path = root / "review" / f"{args.stage}_verified.json"
    require(
        not result_path.exists() and not review_path.exists(), "025 stage cannot overwrite or resume existing results"
    )
    result = dict(
        schema_version="s2a.frozen-readout-owner.v1",
        stage=args.stage,
        prepared_ref=reference(args.prepared),
        source_peer=reference(args.source_peer),
        complete=False,
        passed=False,
        owned_commands=[],
        external_cosmos_before=before,
        automatic_retry=False,
    )
    try:
        if args.stage == "fit":
            entry = owned_command(worker, worker_environment(prepared, False), directory / "producer")
        else:
            entry = run_gpu(prepared, peer, worker, args.stage, directory)
            result["gpu_release"] = reference(directory / "gpu_after.json")
        result["owned_commands"].append(entry)
        completed(result_path, args.prepared)
        source_guard(prepared, peer)
        entry = owned_command(audit, worker_environment(prepared, False), directory / "independent")
        result["owned_commands"].append(entry)
        verified = admitted_audit(review_path, args.stage, args.prepared, result_path)
        after = source_guard(prepared, peer)
        require(after == before, "025 external tracked source identity changed")
        result.update(
            complete=True,
            passed=True,
            producer_result=reference(result_path),
            independent_verification=reference(review_path),
            external_cosmos_after=after,
        )
        if args.stage == "fit":
            result["eligible_for_sampling"] = bool(
                read(result_path)["eligible_for_sampling"] and verified["eligible_for_sampling"]
            )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        write_once(directory / "result.json", result)
    print(json.dumps(dict(stage=args.stage, result=reference(directory / "result.json"))))


def interrupted(signum, frame):
    raise InterruptedError(f"025 owner received signal{signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("features", "fit", "sampling"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--source-peer", type=Path, required=True)
    parser.add_argument("--source-peer-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    execute(args)


if __name__ == "__main__":
    main()
