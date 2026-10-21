"""Prepare or own022's48 exact reproductions, then48 fixed command queries."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import requests
from run_wam_long_service import FINGERPRINT, weight_stats
from run_wam_short_replan import cpu_environment, interrupted, process_record, verify_hashes, write_new

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import owned_child

REGISTRATION = "e117243"
PLAN = "plans/v0.0.1/022_static_command_state_20261021.md"
ROOT = REPO / "outputs/wam_stability_20261021/static_command_state"
BASELINE = REPO / "outputs/wam_stability_20261021/earlier_screening/8400_g10"
CONFIGS = {mode: REPO / f"configs/wam_static_command_{mode}_20261021.json" for mode in ("reproduction", "expanded")}
SOURCES = (
    "plans/v0.0.1/022_static_command_condition_addendum_20261021.md",
    "scripts/run_wam_static_command.py",
    "scripts/serve_wam_static_command.py",
    "scripts/audit_wam_static_command_preparation.py",
    "scripts/run_wam_state_matrix.py",
    "scripts/run_wam_short_replan.py",
    "scripts/run_wam_long_service.py",
    "src/s2a/worker_handshake.py",
    "src/s2a/artifacts.py",
    "src/s2a/wam_runtime.py",
    "src/s2a/checkpoint_audit.py",
    "src/s2a/geometry.py",
    "src/s2a/prediction_audit.py",
    "configs/libero_wam_effect_20261019.json",
)


def static_prompts(cases):
    """Select each task's first registered command, never its phase or outcome."""
    commands = {}
    for sample in cases:
        commands.setdefault(sample["task"], json.loads(sample["published_prompts"]["40"])["command"])
    if len(commands) != 3 or any(not isinstance(value, str) or not value for value in commands.values()):
        raise ValueError("requires three nonempty fixed task commands")
    return {
        task: json.dumps(dict(task_title=task, command=command), ensure_ascii=False)
        for task, command in commands.items()
    }


def same_actions(current, original):
    """Compare full40×128 raw and normalized values, excluding output metadata."""
    new = current["outputs"][0]["content"]
    old = original["outputs"][0]["content"]
    for name, a, b in (
        ("raw", np.asarray(new["action"]), np.asarray(old["action"])),
        (
            "normalized",
            np.asarray(new["action128"]["normalized_action"]),
            np.asarray(old["action128"]["normalized_action"]),
        ),
    ):
        if a.shape != (40, 128) or b.shape != a.shape or not np.array_equal(a, b):
            raise ValueError(f"022 full128 reproduction differs: {name}")


def preparation_only_root(samples_by_mode, configurations):
    """Allow another immutable preparation only while no inference has begun."""
    previous = []
    if ROOT.exists():
        for entry in ROOT.iterdir():
            if entry.name in configurations:
                assert entry.is_dir() and sorted(p.name for p in entry.iterdir()) == ["samples"]
                assert sorted(p.name for p in (entry / "samples").iterdir()) == ["manifest.json"]
                assert json.loads((entry / "samples/manifest.json").read_text()) == samples_by_mode[entry.name]
            else:
                assert entry.is_file() and entry.name.startswith("prepared") and entry.suffix == ".json"
                old = json.loads(entry.read_text())
                assert old["schema_version"] == "s2a.static-command-preparation.v1"
                assert old["model_started"] is False and old["physics_started"] is False
                previous.append(dict(path=str(entry), sha256=sha256(entry)))
    return sorted(previous, key=lambda item: item["path"])


def prepare(output_path):
    assert output_path.parent == ROOT and output_path.name.startswith("prepared") and output_path.suffix == ".json"
    assert not output_path.exists()
    signatures = {}
    for relative in (PLAN, *(str(path.relative_to(REPO)) for path in CONFIGS.values())):
        frozen = subprocess.check_output(["git", "show", f"{REGISTRATION}:{relative}"], cwd=REPO)
        assert sha256(REPO / relative) == hashlib.sha256(frozen).hexdigest()
        signatures[str(REPO / relative)] = sha256(REPO / relative)
    for relative in SOURCES:
        signatures[str(REPO / relative)] = sha256(REPO / relative)
    source = BASELINE / "samples/manifest.json"
    baseline = json.loads(source.read_text())
    assert len(baseline["cases"]) == 24 and len({x["episode"] for x in baseline["cases"]}) == 6
    assert len({(x["episode"], x["anchor_frame"]) for x in baseline["cases"]}) == 22
    assert all(x["split"] == "diagnostic" for x in baseline["cases"])
    signatures[str(source)] = sha256(source)
    prompts = static_prompts(baseline["cases"])
    configurations = {mode: json.loads(path.read_text()) for mode, path in CONFIGS.items()}
    requests_list = []
    samples_by_mode = {}
    for mode, config in configurations.items():
        omit = {"output_dir", "notes", "checkpoints", "offline_prompt_modes"}
        assert {k: v for k, v in config.items() if k not in omit} == {
            k: v for k, v in baseline["config"].items() if k not in omit
        }
        ck = config["checkpoints"][0]
        assert ck == dict(baseline["config"]["checkpoints"][0], gpu=7, port=10047)
        assert config["offline_prompt_modes"] == ["task_only" if mode == "reproduction" else "published_caption"]
        samples = copy.deepcopy(baseline)
        samples["config"] = config
        for sample in samples["cases"]:
            for field in ("arrays", "image"):
                path = Path(sample[f"{field}_path"])
                assert path.is_absolute() and sha256(path) == sample[f"{field}_sha256"]
                signatures[str(path)] = sha256(path)
            if mode == "expanded":
                sample["published_prompts"]["40"] = prompts[sample["task"]]
            for seed in (42, 43):
                old_name = f"{sample['name']}_h40_task_only_s{seed}"
                old_case = BASELINE / "state_queries/8400_masked" / old_name
                old_request = old_case / "request.json"
                old_response = old_case / "response.json"
                request = json.loads(old_request.read_text())
                assert request["seed"] == seed and request["prompt"] == sample["task"]
                assert request["num_steps"] == 35 and request["guidance"] == 1.0 and request["action_chunk_size"] == 40
                if mode == "expanded":
                    request = dict(
                        request,
                        name=old_name.replace("_task_only_", "_published_caption_"),
                        prompt=prompts[sample["task"]],
                    )
                requests_list.append(
                    dict(
                        mode=mode,
                        request=request,
                        original_response=str(old_response),
                        original_request=str(old_request),
                    )
                )
                for path in (old_request, old_response):
                    signatures[str(path)] = sha256(path)
        samples_by_mode[mode] = samples
    assert len(requests_list) == 96
    weights = weight_stats(configurations["reproduction"]["checkpoints"][0])
    assert weights["files_stat_rechecked"] == 163
    signatures[str(REPO / FINGERPRINT)] = sha256(REPO / FINGERPRINT)
    checkpoint = configurations["reproduction"]["checkpoints"][0]
    model_config = Path(checkpoint["run"]) / "config.yaml"
    signatures[str(model_config)] = sha256(model_config)
    cosmos = Path(configurations["reproduction"]["cosmos_repo"])
    for relative in (
        "cosmos_framework/inference/action_serving.py",
        "cosmos_framework/inference/action_server.py",
        "cosmos_framework/inference/args.py",
        "cosmos_framework/data/generator/action/unified_action128.py",
    ):
        path = cosmos / relative
        signatures[str(path)] = sha256(path)
    previous = preparation_only_root(samples_by_mode, configurations)
    ROOT.mkdir(parents=True, exist_ok=True)
    for mode, samples in samples_by_mode.items():
        path = REPO / configurations[mode]["output_dir"] / "samples/manifest.json"
        if not path.exists():
            write_new(path, samples)
        signatures[str(path)] = sha256(path)
    protocol = dict(
        schema_version="s2a.static-command-preparation.v1",
        registration_commit=REGISTRATION,
        root=str(ROOT),
        configs={mode: dict(path=str(path), sha256=sha256(path)) for mode, path in CONFIGS.items()},
        commands=prompts,
        requests=requests_list,
        actual_checkpoint_stat=weights,
        query_budget=dict(reproduction=48, expanded=48, physics_steps=0, fresh_initial_states=0),
        input_sha256=signatures,
        model_started=False,
        physics_started=False,
        previous_immutable_preparations=previous,
    )
    write_new(output_path, protocol)
    print(json.dumps(dict(path=str(output_path), sha256=sha256(output_path), query_budget=protocol["query_budget"])))


def reproduction_admission(protocol):
    signatures = {}
    for item in protocol["requests"][:48]:
        request = item["request"]
        case = ROOT / "reproduction/state_queries/8400_masked" / request["name"]
        response_path = case / "response.json"
        same_actions(json.loads(response_path.read_text()), json.loads(Path(item["original_response"]).read_text()))
        assert json.loads((case / "request.json").read_text()) == request
        for path in (response_path, case / "request.json", Path(item["original_response"])):
            signatures[str(path)] = sha256(path)
    admission = dict(passed=True, verified_queries=48, full128_raw_and_normalized_exact=True, input_sha256=signatures)
    write_new(ROOT / "reproduction_admission.json", admission)
    return admission


def run(prepared_path, prepared_sha, dry_run):
    assert prepared_path.resolve().parent == ROOT and sha256(prepared_path) == prepared_sha
    protocol = json.loads(prepared_path.read_text())
    assert protocol["schema_version"] == "s2a.static-command-preparation.v1"
    verify_hashes(protocol["input_sha256"])
    config = json.loads(CONFIGS["reproduction"].read_text())
    assert weight_stats(config["checkpoints"][0]) == protocol["actual_checkpoint_stat"]
    model_command = [
        str(Path(config["cosmos_repo"]) / ".venv/bin/python"),
        "scripts/serve_wam_static_command.py",
        "--prepared",
        str(prepared_path),
        "--prepared-sha256",
        prepared_sha,
    ]
    if dry_run:
        print(json.dumps(dict(model_command=model_command, physical_gpu=7, port=10047, queries=96, physics_steps=0)))
        return
    directory = ROOT / "orchestration"
    directory.mkdir(exist_ok=False)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    lock_path = REPO / "outputs/wam_stability_20261021/gpu_locks/physical7.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="7",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=config["cosmos_repo"] + ":" + str(REPO / "src"),
        MASTER_PORT="31047",
        COSMOS_TRAINING="0",
        OMP_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        TORCHINDUCTOR_CACHE_DIR=str(ROOT / "model_cache"),
    )
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory = subprocess.check_output(
            ["nvidia-smi", "--id=7", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        )
        assert int(memory.strip()) == 0
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 10047))
        model_folder = directory / "model_ownership"
        model_folder.mkdir()
        with (
            (directory / "service.log").open("x") as stream,
            owned_child(model_command, environment=environment, cwd=REPO, folder=model_folder, stream=stream) as (
                model,
                birth,
            ),
        ):
            write_new(
                directory / "manifest.json",
                dict(
                    prepared=dict(path=str(prepared_path), sha256=prepared_sha),
                    model_command=model_command,
                    coordinator_pid=os.getpid(),
                    model=dict(birth),
                    physical_gpu=7,
                    port=10047,
                    automatic_retry=False,
                    new_physics_steps=0,
                ),
            )
            with requests.Session() as session:
                session.trust_env = False
                deadline = time.monotonic() + 900
                while True:
                    if model.poll() is not None:
                        raise ChildProcessError("022 model exited before readiness")
                    try:
                        response = session.get("http://127.0.0.1:10047/info", timeout=2)
                        response.raise_for_status()
                        info = response.json()
                        break
                    except requests.RequestException:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("022 model readiness timed out") from None
                        time.sleep(2)
            expected = Path(config["checkpoints"][0]["run"]) / "checkpoints/iter_000008400/model"
            assert Path(info["checkpoint_path"]) == expected and not info["ema_weights"]
            assert info["attention_backend"] == "sdpa" and info["acceptance_intervention"]["active_slot_mask"]
            observed = process_record(model.pid, "7", start_ticks=birth["start_ticks"])
            assert observed["present"]
            write_new(directory / "model_ready.json", dict(observed, info=info))
            for mode in ("reproduction", "expanded"):
                verify_hashes(protocol["input_sha256"])
                command = [
                    str(REPO / ".venv/bin/python"),
                    "scripts/run_wam_state_matrix.py",
                    "--config",
                    str(CONFIGS[mode]),
                    "--seed",
                    "42",
                    "--checkpoint",
                    "8400_masked",
                ]
                folder = directory / f"{mode}_ownership"
                folder.mkdir()
                cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
                with (
                    (directory / f"{mode}.log").open("x") as log,
                    owned_child(command, environment=cpu_env, cwd=REPO, folder=folder, stream=log) as (job, _),
                ):
                    if job.wait() != 0:
                        raise ChildProcessError(f"022 {mode} query worker failed; no retry")
                if mode == "reproduction":
                    reproduction_admission(protocol)
            verify_hashes(protocol["input_sha256"])
            assert weight_stats(config["checkpoints"][0]) == protocol["actual_checkpoint_stat"]
        write_new(
            directory / "complete.json",
            dict(
                complete=True,
                reproduction_queries=48,
                expanded_queries=48,
                physics_steps=0,
                fresh_initial_states=0,
                automatic_retry=False,
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--preparation-output", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--prepared-sha256")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cpu_environment()
    if args.prepare:
        assert not args.prepared and not args.prepared_sha256 and not args.dry_run
        prepare((args.preparation_output or ROOT / "prepared.json").absolute())
    else:
        assert args.prepared and args.prepared_sha256 and not args.preparation_output
        run(args.prepared, args.prepared_sha256, args.dry_run)


if __name__ == "__main__":
    main()
