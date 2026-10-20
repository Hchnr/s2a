"""011: one GPU6 service, sequential48-query35/70/140 state-only matrices."""

import argparse
import copy
import datetime
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

from s2a.sampler_trace import check_sources, file_sha, write_new_json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/wam_stability_20261021/sampler_steps_state"
PREPARED = DEST / "orchestration/prepared.json"
ORIGINAL = ROOT / "outputs/wam_stability_20261021/earlier_screening/8400_g10"
FINGERPRINTS = ROOT / "outputs/wam_independent_20261020/checkpoint_fingerprints.json"
PLAN = ROOT / "plans/v0.0.1/011_sampler_steps_state_20261021.md"
CONFIGS = {n: ROOT / f"configs/wam_sampler_steps_{n}_20261021.json" for n in (35, 70, 140)}


def read(path):
    return json.loads(Path(path).read_text())


def ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def check_weights(files):
    for name, item in files.items():
        s = Path(name).stat()
        if s.st_size != item["bytes"] or s.st_mtime_ns != item["mtime_ns"]:
            raise ValueError(f"checkpoint stat changed since byte hashing: {name}")


def configs():
    all_configs = {n: read(p) for n, p in CONFIGS.items()}
    base = {k: v for k, v in all_configs[35].items() if k not in ("notes", "output_dir", "sampling_steps")}
    for n, c in all_configs.items():
        if {k: v for k, v in c.items() if k not in ("notes", "output_dir", "sampling_steps")} != base:
            raise ValueError("sampling configs differ beyond registered steps/output/notes")
        if (
            c["sampling_steps"] != n
            or c["seed"] != 42
            or c["offline_horizons"] != [40]
            or c["offline_prompt_modes"] != ["task_only"]
            or c["sampling_seeds"] != [42, 43]
            or c["guidance"] != 1.0
            or c["shift"] != 5.0
            or c["control_fps"] != 20
            or len(c["checkpoints"]) != 1
            or c["checkpoints"][0]["name"] != "8400_masked"
            or c["checkpoints"][0]["iteration"] != "iter_000008400"
            or c["checkpoints"][0]["gpu"] != 6
            or c["checkpoints"][0]["port"] != 9566
            or (ROOT / c["output_dir"]).resolve() != DEST / f"8400_steps{n}"
        ):
            raise ValueError("configuration is outside registered011 budget/interface")
    return all_configs


def prepare(all_configs):
    if DEST.exists():
        raise FileExistsError("011 output exists; never overwrite or automatically resume")
    c = all_configs[35]
    source = Path(c["source_samples_manifest"])
    if file_sha(source) != "0ffc7115c6af4f1b7c944b2572805956f6db229d2e63d602442e20a8e1cc1049":
        raise ValueError("original sample manifest changed")
    selected = read(ORIGINAL / "samples/manifest.json")
    cases = [x for x in read(source)["cases"] if x["split"] == "diagnostic"]
    if cases != selected["cases"] or len(cases) != 24 or len({x["episode"] for x in cases}) != 6:
        raise ValueError("must retain original24 diagnostic anchors/6 episodes/order")
    anchors = {(x["episode"], x["anchor_frame"]) for x in cases}
    if len(anchors) != 22:
        raise ValueError("original repeated-anchor accounting changed")
    inputs = {
        str(source): file_sha(source),
        str(ORIGINAL / "samples/manifest.json"): file_sha(ORIGINAL / "samples/manifest.json"),
    }
    for sample in cases:
        for kind in ("arrays", "image"):
            p, h = sample[f"{kind}_path"], sample[f"{kind}_sha256"]
            if not Path(p).is_absolute() or file_sha(p) != h:
                raise ValueError("selected source sample changed")
            inputs[p] = h
    cached_root = ORIGINAL / "state_queries/8400_masked"
    complete = read(cached_root / "complete.json")
    if complete["cases"] != 48:
        raise ValueError("cached35 is incomplete")
    cached, order = {}, []
    for sample in cases:
        for seed in (42, 43):
            name = f"{sample['name']}_h40_task_only_s{seed}"
            case = cached_root / name
            request_path, response_path, result_path = (
                case / n for n in ("request.json", "response.json", "result.json")
            )
            request, result = read(request_path), read(result_path)
            if (
                file_sha(result_path) != complete["result_files"][name]
                or file_sha(response_path) != result["response_sha256"]
                or request["seed"] != seed
                or request["num_steps"] != 35
                or request["guidance"] != 1.0
                or request["shift"] != 5.0
                or request["vision_path"] != sample["image_path"]
                or request["prompt"] != sample["task"]
                or request["action_chunk_size"] != 40
            ):
                raise ValueError("cached35 provenance differs")
            for p in (request_path, response_path, result_path):
                inputs[str(p)] = file_sha(p)
            cached[name] = dict(
                request_path=str(request_path),
                request_sha256=file_sha(request_path),
                response_path=str(response_path),
                response_sha256=file_sha(response_path),
                image_sha256=sample["image_sha256"],
            )
            order.append(name)
    if set(order) != set(complete["result_files"]):
        raise ValueError("cached48 coverage differs")
    inputs[str(cached_root / "complete.json")] = file_sha(cached_root / "complete.json")
    inputs[str(cached_root / "manifest.json")] = file_sha(cached_root / "manifest.json")
    ck = c["checkpoints"][0]
    checkpoint = str(Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model")
    weights = next(x for x in read(FINGERPRINTS)["checkpoints"] if x["name"] == "8400_masked")
    if Path(weights["checkpoint"]).resolve() != Path(checkpoint).resolve():
        raise ValueError("checkpoint byte fingerprint identity differs")
    check_weights(weights["files"])
    inputs[str(FINGERPRINTS)] = file_sha(FINGERPRINTS)
    inputs[str(PLAN)] = file_sha(PLAN)
    local = [
        "src/s2a/sampler_trace.py",
        "scripts/serve_wam_sampler_trace.py",
        "scripts/run_wam_sampler_steps.py",
        "tests/test_sampler_trace.py",
        "scripts/run_wam_state_matrix.py",
        "src/s2a/wam_runtime.py",
        "src/s2a/artifacts.py",
        "src/s2a/checkpoint_audit.py",
        "src/s2a/geometry.py",
        "src/s2a/prediction_audit.py",
    ]
    external = [
        "cosmos_framework/model/generator/omni_mot_model.py",
        "cosmos_framework/model/generator/diffusion/samplers/unipc.py",
        "cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py",
        "cosmos_framework/model/generator/diffusion/samplers/utils.py",
        "cosmos_framework/inference/action_server.py",
        "cosmos_framework/inference/action_serving.py",
        "cosmos_framework/inference/args.py",
        "cosmos_framework/inference/inference.py",
        "cosmos_framework/inference/common/init.py",
        "cosmos_framework/utils/misc.py",
    ]
    sources = {str(ROOT / p): file_sha(ROOT / p) for p in local}
    sources.update({str(Path(c["cosmos_repo"]) / p): file_sha(Path(c["cosmos_repo"]) / p) for p in external})
    sources.update({str(p): file_sha(p) for p in CONFIGS.values()})
    sources[c["policy_template"] if Path(c["policy_template"]).is_absolute() else str(ROOT / c["policy_template"])] = (
        file_sha(ROOT / c["policy_template"])
    )
    sources[str(Path(ck["run"]) / "config.yaml")] = file_sha(Path(ck["run"]) / "config.yaml")
    DEST.mkdir()
    (DEST / "orchestration").mkdir()
    snapshots = DEST / "orchestration/source_snapshot"
    snapshots.mkdir()
    copies = {}
    for i, (p, h) in enumerate(sources.items()):
        dest = snapshots / f"{i:03d}_{Path(p).name}"
        shutil.copyfile(p, dest)
        if file_sha(dest) != h:
            raise ValueError("source snapshot differs")
        copies[str(dest)] = h
    for n, config in all_configs.items():
        dest = ROOT / config["output_dir"] / "samples"
        dest.mkdir(parents=True)
        manifest = copy.deepcopy(selected)
        manifest["config"] = config
        write_new_json(dest / "manifest.json", manifest)
        inputs[str(dest / "manifest.json")] = file_sha(dest / "manifest.json")
    record = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        prepared_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        root=str(DEST),
        budget=dict(requests=144, requests_per_stage=48, physics_steps=0),
        diagnostic_phase_cases=24,
        unique_anchors=22,
        unique_anchor_seed_pairs=44,
        configs={str(n): dict(path=str(p), sha256=file_sha(p)) for n, p in CONFIGS.items()},
        checkpoint=checkpoint,
        weight_files=weights["files"],
        source_sha256=sources,
        unipc_source=str(Path(c["cosmos_repo"]) / external[1]),
        input_sha256=inputs,
        snapshot_sha256=copies,
        cached_cases=cached,
        request_order=order,
        physical_gpu=6,
        port=9566,
        full_weight_bytes_rehashed=False,
        registration=dict(
            budget_commit="7a9e571", promotion_commit="1ad56d4", plan_path=str(PLAN), plan_sha256=file_sha(PLAN)
        ),
        source_tree_correction="Prior unimplemented proposal read /share/project/eai_pwm/repos/cosmos-framework-pwm; actual serving sources here follow config.cosmos_repo=/share/project/eai_pwm/home/hcr/repos/cosmos-framework-pwm. Old proposal bytes remain unchanged.",
    )
    write_new_json(PREPARED, record)
    return record


def validate_prepared(protocol, prepared_sha=None):
    if prepared_sha is not None and file_sha(PREPARED) != prepared_sha:
        raise ValueError("prepared protocol bytes changed")
    if protocol["root"] != str(DEST) or protocol["budget"] != dict(
        requests=144, requests_per_stage=48, physics_steps=0
    ):
        raise ValueError("prepared root/budget differs")
    check_sources(protocol["source_sha256"])
    check_sources(protocol["input_sha256"])
    check_sources(protocol["snapshot_sha256"])
    check_weights(protocol["weight_files"])


def stage_admission(protocol, stage):
    out = ROOT / read(CONFIGS[stage])["output_dir"] / "state_queries/8400_masked"
    complete = read(out / "complete.json")
    if complete["cases"] != 48 or set(complete["result_files"]) != set(protocol["request_order"]):
        raise ValueError("stage lacks full registered48 requests")
    hashes, result_hashes = {}, {}
    for name in protocol["request_order"]:
        trace_path = DEST / f"service/traces{stage}/{name}.json"
        trace, result = read(trace_path), read(out / name / "result.json")
        if (
            not trace["complete"]
            or trace["velocity_evaluation_count"] != stage
            or (stage == 35 and not trace["cached35_full128_exact"])
            or (stage != 35 and not trace["actual_state_and_condition_pair_with35"])
            or file_sha(out / name / "response.json") != result["response_sha256"]
            or file_sha(out / name / "result.json") != complete["result_files"][name]
        ):
            raise ValueError("stage raw/trace/result admission differs")
        hashes[name] = file_sha(trace_path)
        result_hashes[name] = complete["result_files"][name]
    return dict(
        passed=True,
        sampling_steps=stage,
        full_cases=48,
        complete_sha256=file_sha(out / "complete.json"),
        trace_sha256=hashes,
        result_sha256=result_hashes,
    )


def stop_owned(process, start_ticks):
    if process is None or process.poll() is not None:
        return
    if ticks(process.pid) != start_ticks:
        raise ValueError("refusing to stop a process with another startticks identity")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if ticks(process.pid) != start_ticks:
            raise ValueError("process identity changed before forced teardown")
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=(42,))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("CPU coordinator requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    all_configs = configs()
    if args.prepare_only and args.dry_run:
        raise ValueError("choose preparation or read-only dry-run")
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    physical_gpu=6,
                    port=9566,
                    stages=[35, 70, 140],
                    requests=144,
                    physics_steps=0,
                    prepared=str(PREPARED),
                    configs={n: str(p) for n, p in CONFIGS.items()},
                )
            )
        )
        return
    if args.prepare_only:
        protocol = prepare(all_configs)
        validate_prepared(protocol)
        print(json.dumps(dict(prepared=str(PREPARED), sha256=file_sha(PREPARED), model_processes_started=0)))
        return
    protocol = read(PREPARED)
    prepared_sha = file_sha(PREPARED)
    validate_prepared(protocol, prepared_sha)
    dest = DEST / "orchestration/run"
    dest.mkdir(exist_ok=False)
    lockdir = ROOT / "outputs/wam_stability_20261021/gpu_locks"
    lockdir.mkdir(exist_ok=True)
    model = job = None
    model_ticks = job_ticks = None
    model_cmd = [
        str(Path(all_configs[35]["cosmos_repo"]) / ".venv/bin/python"),
        "scripts/serve_wam_sampler_trace.py",
        "--prepared",
        str(PREPARED),
        "--prepared-sha256",
        file_sha(PREPARED),
        "--seed",
        "42",
    ]

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"owned011 coordinator received signal{signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    with (lockdir / "physical6.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            used = subprocess.check_output(
                ["nvidia-smi", "--id=6", "--query-gpu=memory.used,uuid", "--format=csv,noheader,nounits"], text=True
            ).strip()
            if int(used.split(",")[0]) >= 500:
                raise ValueError("allowed GPU6 is occupied; no task was stopped")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 9566))
            write_new_json(
                dest / "launch.json",
                dict(
                    head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    coordinator_pid=os.getpid(),
                    coordinator_startticks=ticks(os.getpid()),
                    prepared_sha256=file_sha(PREPARED),
                    physical_gpu=6,
                    gpu_uuid=used.split(",")[1].strip(),
                    port=9566,
                    model_command=model_cmd,
                    budget=protocol["budget"],
                ),
            )
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="6",
                LD_LIBRARY_PATH="",
                MASTER_PORT="31966",
                COSMOS_TRAINING="0",
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=all_configs[35]["cosmos_repo"] + ":" + str(ROOT / "src"),
                OMP_NUM_THREADS="2",
                TOKENIZERS_PARALLELISM="false",
                TORCHINDUCTOR_CACHE_DIR=str(DEST / "model_cache"),
            )
            with (dest / "service.log").open("x") as stream:
                model = subprocess.Popen(
                    model_cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
                )
            model_ticks = ticks(model.pid)
            write_new_json(
                dest / "model_process.json",
                dict(pid=model.pid, start_ticks=model_ticks, physical_gpu=6, command=model_cmd),
            )
            import requests

            with requests.Session() as session:
                session.trust_env = False
                for _ in range(180):
                    if model.poll() is not None:
                        raise ValueError("owned trace model exited during startup")
                    try:
                        response = session.get("http://127.0.0.1:9566/info", timeout=2)
                        response.raise_for_status()
                        info = response.json()
                        break
                    except requests.RequestException:
                        time.sleep(5)
                else:
                    raise TimeoutError("trace service readiness timed out")
            if (
                info["checkpoint_path"] != protocol["checkpoint"]
                or info["ema_weights"]
                or not info["sampler_trace_protocol"]["active_slot_mask"]
            ):
                raise ValueError("private service identity differs")
            write_new_json(dest / "server_info.json", info)
            for stage in (35, 70, 140):
                validate_prepared(protocol, prepared_sha)
                cmd = [
                    str((ROOT / ".venv/bin/python").absolute()),
                    "scripts/run_wam_state_matrix.py",
                    "--config",
                    str(CONFIGS[stage]),
                    "--seed",
                    "42",
                    "--checkpoint",
                    "8400_masked",
                ]
                with (dest / f"steps{stage}.log").open("x") as stream:
                    job = subprocess.Popen(
                        cmd,
                        cwd=ROOT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                job_ticks = ticks(job.pid)
                write_new_json(
                    dest / f"steps{stage}_process.json", dict(pid=job.pid, start_ticks=job_ticks, command=cmd)
                )
                if job.wait() != 0:
                    raise ValueError(f"steps{stage} matrix failed; no automatic retry or later stage")
                validate_prepared(protocol, prepared_sha)
                admission = stage_admission(protocol, stage)
                write_new_json(dest / f"admission{stage}.json", admission)
            write_new_json(
                dest / "complete.json",
                dict(
                    complete=True,
                    requests=144,
                    physics_steps=0,
                    admissions={str(n): file_sha(dest / f"admission{n}.json") for n in (35, 70, 140)},
                ),
            )
        except BaseException as exc:
            write_new_json(
                dest / "error.json", dict(error=repr(exc), no_automatic_retry=True, scientific_failure_inferred=False)
            )
            raise
        finally:
            try:
                stop_owned(job, job_ticks)
            finally:
                stop_owned(model, model_ticks)
                write_new_json(
                    dest / "release.json",
                    dict(
                        model_pid=model.pid if model else None,
                        model_startticks=model_ticks,
                        model_returncode=model.poll() if model else None,
                        last_job_pid=job.pid if job else None,
                        last_job_returncode=job.poll() if job else None,
                    ),
                )


if __name__ == "__main__":
    main()
