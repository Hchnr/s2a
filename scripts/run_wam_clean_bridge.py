"""Own the finite035 WAM clean bridge qualification, without fitting or physics."""

import argparse
import copy
import datetime
import fcntl
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import tarfile
import time
from pathlib import Path

from run_native_motor_hold_diagnostic import LOCATOR
from run_wam_readout import owned_command

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import write_once

CONFIG = REPO / "configs/wam_clean_bridge_20261022.json"
PLAN = REPO / "plans/v0.0.1/035_clean_bridge_qualification_20261022.md"
SOURCE_COMMIT = "8ec9484b9a6dadb9467b990ae89e80d604d47224"
SOURCE_REPOSITORY = "/share/project/eai_pwm/repos/cosmos-framework-pwm"
GPU_IDENTITIES = {
    "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f": (2, 6),
    "GPU-7a163b30-e2cd-f747-6075-3265058ad0b8": (3, 7),
}
BUDGET = dict(requests=7, denoise_calls=245, model_initializations=1, fit_groups=0, physics_steps=0, fresh_cases=0)
SCHEMA = "s2a.clean-bridge-preparation.v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def resolve_path(path):
    return LOCATOR.resolve_path(path)


def reference(path):
    return LOCATOR.reference(path)


def read(path, digest=None):
    return LOCATOR.read(path if isinstance(path, dict) else dict(path=str(path), sha256=digest))


ref, resolve = reference, resolve_path


def publish(path, value):
    write_once(path, value)
    return reference(path)


def cpu_owner():
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "035 owner requires S2A cwd and explicitly empty CUDA/LD",
    )


def validate_config(config):
    require(config["seed"] == 42 and config["budget"] == BUDGET, "035 fixed request/NFE budget changed")
    require(
        all(
            config.get(key) is False
            for key in ("model_training_authorized", "fresh_performance_authorized", "automatic_retry")
        ),
        "035 does not authorize fitting, fresh performance, or automatic retry",
    )
    require(config["gpu_uuid"] in GPU_IDENTITIES, "035 GPU must be one of the two allowed UUIDs")
    require(
        config["allowed_gpu_uuids"] == list(GPU_IDENTITIES)
        and config["source_repository"] == SOURCE_REPOSITORY
        and config["source_commit"] == SOURCE_COMMIT,
        "035 allowed GPU/source contract changed",
    )
    root = (REPO / config["output_dir"]).resolve()
    require(root.is_relative_to(REPO / "outputs"), "035 output must remain within S2A outputs")
    require(Path(config["cosmos_source_root"]) == root / "source/cosmos", "035 immutable Cosmos root changed")
    require(config["python_version"] == [3, 13, 14] and config["master_port"] == 31335, "035 runtime contract changed")
    require(config["historical_generation_indices"] == [0, 16, 32], "035 historical request roster changed")
    roster = config["noise_roster"]
    require(
        roster["evaluation_seeds"] == [42, 43]
        and isinstance(roster["training_seeds"], list)
        and bool(roster["training_seeds"])
        and all(type(seed) is int and seed >= 0 and seed not in (42, 43) for seed in roster["training_seeds"]),
        "035 training/evaluation noise rosters must remain disjoint",
    )
    residual = config["nonzero_clean_residual_active10"]
    require(
        config["feature_terms"] == []
        and config["mask_policy"] == "native_and_registered_active_slot_wrapper"
        and isinstance(residual, list)
        and len(residual) == 10
        and all(type(x) in (float, int) and math.isfinite(x) and 0 < abs(x) <= 1e-4 for x in residual),
        "035 explicit feature/mask/nonzero qualification contract changed",
    )
    return root


def check_hashes(mapping):
    require(isinstance(mapping, dict) and bool(mapping), "035 source/input SHA mapping must be nonempty")
    for path, digest in mapping.items():
        require(sha256(resolve_path(path)) == digest, f"035 frozen source/input changed: {path}")


def weight_stats(mapping):
    require(isinstance(mapping, dict) and len(mapping) == 163, "035 historical8400 file roster must contain163 files")
    for path, value in mapping.items():
        stat = resolve_path(path).stat()
        require((stat.st_size, stat.st_mtime_ns) == (value["bytes"], value["mtime_ns"]), "035 weight stat changed")


def historical_requests(config, historical):
    requests = []
    images = {item["name"]: item for item in historical["anchors"]}
    for index in config["historical_generation_indices"]:
        entry = historical["free_generations"][index]
        require(entry["variant"] == "raw" and entry["sampling_seed"] == 42, "035 requires original raw seed42 requests")
        anchor = images[entry["anchor_name"]]
        historical_image_ref = dict(path=anchor["image_path"], sha256=anchor["image_sha256"])
        require(
            reference(historical_image_ref["path"]) == historical_image_ref, "035 historical current image SHA changed"
        )
        image_ref = reference(resolve_path(historical_image_ref["path"]))
        request = copy.deepcopy(entry["request"])
        require(request["vision_path"] == historical_image_ref["path"], "035 historical image declarations differ")
        request["vision_path"] = image_ref["path"]
        require(
            request["seed"] == 42
            and request["model_mode"] == "wam"
            and request["guidance"] == 1.0
            and request["shift"] == 5.0
            and request["num_steps"] == 35
            and request["action_chunk_size"] == 40
            and anchor["anchor_frame"] == 0,
            "035 original fixed inference request contract changed",
        )
        historical_expected = entry["cache"]["response"]
        require(reference(historical_expected["path"]) == historical_expected, "035 historical response bytes changed")
        expected = reference(resolve_path(historical_expected["path"]))
        requests.append(
            dict(
                name=entry["name"],
                request=request,
                expected_response_ref=expected,
                image_ref=image_ref,
                historical_image_ref=historical_image_ref,
                historical_entry=entry,
            )
        )
    require(len(requests) == 3 and len({item["name"] for item in requests}) == 3, "035 three-request roster differs")
    require(config["nonzero_request_name"] == requests[0]["name"], "035 nonzero request must be the fixed first task")
    return requests


def external_identity(config):
    repo = config["source_repository"]
    head = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"], text=True)
    require(
        head == SOURCE_COMMIT and not dirty.strip(),
        "035 external tracked source must remain the registered clean commit",
    )
    return dict(repository=repo, head=head, tracked_source_clean=True)


def freeze_cosmos_source(config, root):
    before = external_identity(config)
    target = root / "source/cosmos"
    target.mkdir(parents=True, exist_ok=False)
    archive = root / "source/cosmos.tar"
    subprocess.run(
        ["git", "-C", config["source_repository"], "archive", "--format=tar", f"--output={archive}", SOURCE_COMMIT],
        check=True,
    )
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            require(
                not Path(member.name).is_absolute()
                and ".." not in Path(member.name).parts
                and (member.isfile() or member.isdir()),
                "035 source archive contains a link or unsafe entry",
            )
        stream.extractall(target, filter="data")
    require(external_identity(config) == before, "035 external source changed while being archived")
    sources = {
        str(path): sha256(path)
        for path in sorted(target.rglob("*"))
        if path.is_file() and path.suffix in (".py", ".yaml", ".yml", ".json", ".toml")
    }
    require(bool(sources), "035 frozen Cosmos source is empty")
    return sources, dict(**before, archive_ref=reference(archive))


def read_prepared(path, digest=None):
    prepared = read(path, digest)
    require(prepared["schema_version"] == SCHEMA, "035 preparation schema changed")
    root = validate_config(prepared["config"])
    has_bound_sources = "config_ref" in prepared or "plan_ref" in prepared
    if has_bound_sources:
        require(
            "config_ref" in prepared and "plan_ref" in prepared, "035 config/plan references must be bound together"
        )
        for name in ("config_ref", "plan_ref"):
            bound = prepared[name]
            require(
                isinstance(bound, dict) and set(bound) == {"path", "sha256"}, "035 invalid bound config/plan reference"
            )
            source = Path(bound["path"]).resolve()
            require(source.is_relative_to(REPO), "035 bound config/plan must remain within S2A")
            snapshot = root / "source/s2a" / source.relative_to(REPO)
            require(
                prepared["source_sha256"].get(str(source)) == bound["sha256"]
                and prepared["snapshot_sha256"].get(str(snapshot)) == bound["sha256"],
                "035 bound config/plan source or snapshot was not frozen",
            )
        current_config = read(prepared["config_ref"])
        require(reference(prepared["plan_ref"]["path"]) == prepared["plan_ref"], "035 bound plan changed")
    else:
        # Historical preparations retain their original default-config contract.
        current_config = json.loads(CONFIG.read_text())
    require(
        Path(prepared["root"]) == root and current_config == prepared["config"],
        "035 current config/root changed",
    )
    for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
        check_hashes(prepared[key])
    require(str(Path(__file__).resolve()) in prepared["source_sha256"], "035 actual owner source was not frozen")
    weight_stats(prepared["weight_files"])
    historical = read(prepared["config"]["historical_prepared_ref"])
    require(
        prepared["requests3"] == historical_requests(prepared["config"], historical),
        "035 historical request/image roster changed",
    )
    require(
        prepared["weight_files"] == historical["weight_files"], "035 historical weight stat/claimed SHA roster changed"
    )
    small = {
        path: value["sha256"]
        for path, value in historical["weight_files"].items()
        if Path(path).name in ("config.yaml", ".metadata", ".save_complete")
    }
    require(
        len(small) == 3
        and prepared["checkpoint_small_metadata_sha256"] == small
        and all(prepared["input_sha256"].get(path) == digest for path, digest in small.items()),
        "035 all three small checkpoint metadata SHA inputs must be bound",
    )
    for key, name in (("sampler_source_ref", "unipc.py"), ("scheduler_conversion_source_ref", "fm_solvers_unipc.py")):
        expected = root / "source/cosmos/cosmos_framework/model/generator/diffusion/samplers" / name
        require(prepared[key] == reference(expected), "035 actual frozen sampler/conversion source changed")
        require(str(expected) in prepared["source_sha256"], "035 actual sampler source omitted")
    require(
        prepared["weight_integrity_contract"] == "historical163_stat_and_three_small_metadata_sha_no_new_shard_hash",
        "035 weight integrity evidence contract changed",
    )
    return prepared


def prepare(config_path=None, plan_path=None):
    cpu_owner()
    config_path = Path(CONFIG if config_path is None else config_path).resolve()
    plan_path = Path(PLAN if plan_path is None else plan_path).resolve()
    require(
        all(path.is_relative_to(REPO) and path.is_file() for path in (config_path, plan_path)),
        "035 preparation config/plan must be existing files within S2A",
    )
    config_ref, plan_ref = reference(config_path), reference(plan_path)
    config = read(config_ref)
    root = validate_config(config)
    require(not root.exists(), "035 refuses existing preparation/output; no overwrite or automatic retry")
    require(Path(config["model_python"]).is_file(), "035 configured model venv interpreter is unavailable")
    historical = read(config["historical_prepared_ref"])
    requests = historical_requests(config, historical)
    weight_stats(historical["weight_files"])
    inputs = {config["historical_prepared_ref"]["path"]: config["historical_prepared_ref"]["sha256"]}
    for entry in requests:
        for key in ("image_ref", "expected_response_ref"):
            inputs[entry[key]["path"]] = entry[key]["sha256"]
    small_metadata = {}
    for path, value in historical["weight_files"].items():
        if Path(path).name in ("config.yaml", ".metadata", ".save_complete"):
            actual = sha256(resolve_path(path))
            require(actual == value["sha256"], "035 small checkpoint metadata SHA changed")
            inputs[path] = actual
            small_metadata[path] = actual
    require(len(small_metadata) == 3, "035 requires all three checkpoint metadata SHA records")
    root.mkdir(parents=True, exist_ok=False)
    sources, external = freeze_cosmos_source(config, root)
    snapshots = dict(sources)
    own_sources = sorted(
        {p for folder in ("src", "scripts", "tests") for p in (REPO / folder).rglob("*.py")}
        | {config_path, plan_path, REPO / "pyproject.toml"}
    )
    for source in own_sources:
        target = root / "source/s2a" / source.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources[str(source)] = sha256(source)
        snapshots[str(target)] = sha256(target)
        require(sources[str(source)] == snapshots[str(target)], "035 S2A source changed while copied")
    (root / "preparation").mkdir()
    publish(root / "preparation/source_archive.json", external)
    samplers = root / "source/cosmos/cosmos_framework/model/generator/diffusion/samplers"
    prepared_ref = publish(
        root / "preparation/prepared.json",
        dict(
            schema_version=SCHEMA,
            root=str(root),
            config=config,
            config_ref=config_ref,
            plan_ref=plan_ref,
            source_sha256=sources,
            snapshot_sha256=snapshots,
            input_sha256=inputs,
            weight_files=historical["weight_files"],
            checkpoint_small_metadata_sha256=small_metadata,
            weight_integrity_contract="historical163_stat_and_three_small_metadata_sha_no_new_shard_hash",
            requests3=requests,
            sampler_source_ref=reference(samplers / "unipc.py"),
            scheduler_conversion_source_ref=reference(samplers / "fm_solvers_unipc.py"),
            cosmos_archive=external,
        ),
    )
    read_prepared(prepared_ref)
    print(json.dumps(prepared_ref), flush=True)
    return prepared_ref


def resource_observation(uuid, *, driver_root=Path("/proc/driver/nvidia/gpus")):
    require(uuid in GPU_IDENTITIES, "035 GPU must be one of the two allowed UUIDs")
    expected_visible, expected_minor = GPU_IDENTITIES[uuid]
    rows = []
    for path in sorted(driver_root.glob("*/information")):
        fields = dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if fields.get("GPU UUID") == uuid:
            rows.append(fields)
    require(len(rows) == 1 and int(rows[0]["Device Minor"]) == expected_minor, "035 driver UUID/device minor differs")
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={uuid}",
            "--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    visible, actual_uuid, bus, memory, utilization = [item.strip() for item in raw.split(",")]
    require(
        int(visible) == expected_visible
        and actual_uuid == uuid
        and bus.lower().lstrip("0") == rows[0]["Bus Location"].lower().lstrip("0"),
        "035 nvidia-smi index/UUID/PCI differs from driver identity",
    )
    raw_apps = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={uuid}",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    apps = []
    for line in raw_apps.splitlines():
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 3, "035 compute-process observation is malformed")
        if fields[1] == uuid:
            apps.append(line)
    return dict(
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        uuid=uuid,
        visible_gpu=int(visible),
        device_minor=expected_minor,
        pci_bus_id=bus,
        memory_mib=int(memory),
        utilization_gpu=int(utilization),
        compute_apps=apps,
        raw_query=raw,
        raw_compute_query=raw_apps,
    )


def wait_release(directory, uuid, *, timeout=60):
    require(0 < timeout <= 60, "035 release observation must be bounded to at most60seconds")
    started = time.monotonic()
    with (directory / "resource_observations.jsonl").open("x") as stream:
        while True:
            observed = resource_observation(uuid)
            observed["seconds_since_actual_owned_wait"] = time.monotonic() - started
            stream.write(json.dumps(observed) + "\n")
            stream.flush()
            if observed["memory_mib"] == 0 and not observed["compute_apps"]:
                observed["released"] = True
                return publish(directory / "gpu_after.json", observed)
            if time.monotonic() - started >= timeout:
                raise TimeoutError("035 owned model exited; GPU release not observed within60seconds")
            time.sleep(0.25)


def environment(prepared, *, gpu):
    config, root = prepared["config"], Path(prepared["root"])
    require(config["gpu_uuid"] in GPU_IDENTITIES, "035 forbidden GPU UUID")
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=config["gpu_uuid"] if gpu else "",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(root / "source/cosmos") + ":" + str(REPO / "src") + ":" + str(REPO / "scripts"),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="31335",
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
        LOCAL_WORLD_SIZE="1",
        COSMOS_TRAINING="0",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        XDG_CACHE_HOME=str(root / "model_cache"),
        HF_HOME=str(root / "model_cache/hf"),
        TORCH_HOME=str(root / "model_cache/torch"),
        TORCHINDUCTOR_CACHE_DIR=str(root / "model_cache/inductor"),
        CUDA_CACHE_PATH=str(root / "model_cache/cuda"),
        NUMBA_CACHE_DIR=str(root / "model_cache/numba"),
        MPLCONFIGDIR=str(root / "model_cache/mpl"),
    )


def planned_commands(prepared, prepared_ref):
    root = Path(prepared["root"])
    common = ["--prepared", prepared_ref["path"], "--prepared-sha256", prepared_ref["sha256"], "--seed", "42"]
    worker = [
        prepared["config"]["model_python"],
        "scripts/qualify_wam_clean_bridge.py",
        *common,
        "--gpu-uuid",
        prepared["config"]["gpu_uuid"],
    ]
    checker = [
        str(REPO / ".venv/bin/python"),
        "scripts/verify_wam_clean_bridge.py",
        *common,
        "--qualification",
        str(root / "qualification/result.json"),
        "--output",
        str(root / "review/verified.json"),
    ]
    return worker, checker


def check_owned_command(entry):
    launch, exit_record, release = (read(entry[key]) for key in ("launch", "exit", "release"))
    require(
        exit_record == dict(returncode=0, actual_wait=True, automatic_retry=False),
        "035 command lacks successful actual wait",
    )
    require(
        all(release[key] == launch["child"][key] for key in ("pid", "start_ticks"))
        and release["reaped"] is True
        and release["returncode"] == 0
        and release["original_error"] is None
        and release["signal_sent"] is False
        and release["only_owned_child_signalled"] is True,
        "035 owned command was not normally reaped without a signal",
    )


def execute(prepared_ref, *, dry_run=False):
    cpu_owner()
    prepared = read_prepared(prepared_ref)
    root = Path(prepared["root"])
    worker, checker = planned_commands(prepared, prepared_ref)
    if dry_run:
        result = dict(
            worker=worker, checker=checker, model_environment=environment(prepared, gpu=True), new_model_calls=0
        )
        # Avoid printing inherited environment values; expose only the selected
        # nonsecret execution contract for review.
        result["model_environment"] = {
            key: result["model_environment"][key]
            for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PYTHONPATH", "MASTER_PORT")
        }
        print(json.dumps(result), flush=True)
        return result
    require(
        not any((root / name).exists() for name in ("qualification", "review", "orchestration")),
        "035 refuses prior stage outputs/retry",
    )
    directory = root / "orchestration"
    directory.mkdir()
    result = dict(
        schema_version="s2a.clean-bridge-owner.v1",
        complete=False,
        passed=False,
        bridge_qualified=False,
        prepared_ref=prepared_ref,
        budget=BUDGET,
        owned_commands=[],
        automatic_retry=False,
        model_training_authorized=False,
        fresh_performance_authorized=False,
    )
    try:
        uuid = prepared["config"]["gpu_uuid"]
        lock_path = REPO / "outputs/wam_stability_20261021/gpu_locks" / f"{uuid}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = resource_observation(uuid)
            publish(directory / "gpu_before.json", before)
            require(
                before["memory_mib"] == 0 and not before["compute_apps"] and before["utilization_gpu"] == 0,
                "035 selected GPU occupied; no existing process changed",
            )
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 31335))
            read_prepared(prepared_ref)
            print(json.dumps(dict(stage="model_start", gpu_uuid=uuid, budget=BUDGET)), flush=True)
            entry = owned_command(worker, environment(prepared, gpu=True), directory / "model")
            result["owned_commands"].append(entry)
            check_owned_command(entry)
            result["gpu_after_ref"] = wait_release(directory, uuid)
            result["resource_observations_ref"] = reference(directory / "resource_observations.jsonl")
        read_prepared(prepared_ref)
        print(json.dumps(dict(stage="independent_start", new_model_calls=0)), flush=True)
        entry = owned_command(checker, environment(prepared, gpu=False), directory / "independent", timeout=1800)
        result["owned_commands"].append(entry)
        check_owned_command(entry)
        audit_ref = reference(root / "review/verified.json")
        audit = read(audit_ref)
        require(
            audit["schema_version"] == "s2a.clean-bridge-independent-verification.v1"
            and audit["passed"] is audit["complete"] is True
            and audit["prepared_ref"] == prepared_ref
            and audit["qualification_ref"] == reference(root / "qualification/result.json")
            and audit["model_training_authorized"] is audit["fresh_performance_authorized"] is False
            and type(audit["bridge_qualified"]) is bool
            and all(audit[key] == value for key, value in BUDGET.items()),
            "035 independent qualification protocol was not complete",
        )
        read_prepared(prepared_ref)
        result.update(complete=True, passed=True, bridge_qualified=audit["bridge_qualified"], review_ref=audit_ref)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        publish(directory / "result.json", result)
    print(json.dumps(reference(directory / "result.json")), flush=True)
    return result


def interrupted(signum, frame):
    raise InterruptedError(f"035 owner interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run"), required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--prepared-sha256")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    if args.stage == "prepare":
        require(not args.dry_run, "035 prepare has no model side effects; dry-run applies only to run")
        prepare(config_path=args.config, plan_path=args.plan)
    else:
        require(args.config is None and args.plan is None, "035 run uses only its prepared config/plan references")
        require(
            args.prepared is not None and args.prepared_sha256 is not None,
            "035 run requires an exact prepared reference",
        )
        execute(dict(path=str(Path(args.prepared).resolve()), sha256=args.prepared_sha256), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
