"""Admit and execute two registered saved-prefix CPU replays."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment, strict_native_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.infrastructure.repository_paths import repository_root
from s2a.research.native_runtime.development_prefix_replay import map_record_xml, saved_prefix

REPO = repository_root(anchor=__file__)
CONFIG = REPO / "configs/tracking/joint_control/multicase_native_prefix_replay_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/multicase_native_prefix_replay_20261110.md"
CORE = REPO / "src/s2a/research/native_runtime/development_prefix_replay.py"
RUNTIME = REPO / "src/s2a/research/native_runtime/development_prefix_runtime.py"
TEST = REPO / "tests/research/native_runtime/test_development_prefix_replay.py"
QA = REPO / "outputs/qa/multicase_native_prefix_replay_20261110/result.json"
DRY = REPO / "outputs/multicase_native_prefix_replay_dry_20261110/result.json"
OUTPUT = REPO / "outputs/multicase_native_prefix_replay_20261110"
REVIEW = {
    "path": str(REPO / "outputs/tracking_wam_prefetch_review_execution_20261028/verified.json"),
    "sha256": "c5e74dd75394dd8ec46b328a8cd1f55f20aa3893e7d08f0becb67bd23eb6cccb",
}
CONTEXT = {
    "path": str(REPO / "outputs/multicase_joint_limit_context_20261110/result.json"),
    "sha256": "50861e33b9fb5727cc174ea04b79dd38a6b02206d6f181b0f56cecb6b5ea6d10",
}
BOOTSTRAP = {
    "path": str(REPO / "outputs/native_runtime_bootstrap_v3_20261029/result.json"),
    "sha256": "28b6001d9d7b2f9d1618968856612479b2f4c623c27979f0f934c3ba621d319c",
}
MIGRATED_MANIFEST = {
    "path": str(REPO / "outputs/native_runtime_path_snapshot_20261029/runtime_manifest.json"),
    "sha256": "ec8c377eb9fe505d69745f55e33b4ccb6f352fdf6d2e49555d6146c39cb111f5",
}
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LP_NUM_THREADS")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    path = Path(path)
    require(path.is_absolute() and path.is_relative_to(REPO), "Input must be lexically inside the repository")
    path = path.resolve()
    require(path.is_relative_to(REPO) and path.is_file(), "Input must resolve to a repository file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def checked(ref, seen=None):
    value = reference(ref["path"])
    require(value["sha256"] == ref["sha256"], f"Changed input: {ref['path']}")
    if seen is not None:
        require(value["path"] not in seen or seen[value["path"]] == value["sha256"], "Conflicting input identity")
        seen[value["path"]] = value["sha256"]
    return Path(value["path"])


def read(ref, seen=None):
    return json.loads(checked(ref, seen).read_text())


def environment():
    strict_native_environment()
    require(all(os.environ.get(key) == "1" for key in THREAD_VARS), "Single-thread CPU numerical libraries")


def read_recorded_prefix(path, count, total):
    """Count the whole bound log without decoding observations beyond the prefix."""
    values = []
    seen = 0
    with Path(path).open() as stream:
        for line in stream:
            if seen < count:
                values.append(json.loads(line))
            seen += 1
    require(seen == total and len(values) == count, "Complete original log and bounded prefix")
    return values


def select_prefix(case_spec, prefix_steps):
    """Select an explicitly bounded earlier endpoint within a reviewed prefix."""
    if prefix_steps is None:
        return dict(case_spec)
    index = case_spec["case_index"]
    require(index in prefix_steps, "A prefix must be specified for every registered case")
    n = prefix_steps[index]
    require(type(n) is int and 0 < n <= case_spec["prefix_steps"], "Earlier prefix must fit reviewed history")
    return dict(case_spec, prefix_steps=n)


def load_inputs(prefix_steps=None):
    refs = {}
    spec_ref = reference(CONFIG)
    spec = read(spec_ref, refs)
    expected = [
        {"case_index": 2, "case_id": "task0_init0_s43_raw", "prefix_steps": 114},
        {"case_index": 6, "case_id": "task1_init0_s43_raw", "prefix_steps": 99},
    ]
    require(
        spec["schema"] == "s2a.multicase-native-prefix-replay.v1" and spec["cases"] == expected,
        "Exact two registered development prefixes",
    )
    require(
        prefix_steps is None or (isinstance(prefix_steps, dict) and set(prefix_steps) == {2, 6}),
        "Explicit earlier endpoints must cover exactly the two reviewed cases",
    )
    require(
        spec["seed"] == 43
        and spec["atol"] == 1e-10
        and spec["rtol"] == 0
        and spec["max_control_steps"] == 213
        and spec["max_native_steps"] == 5325
        and spec["world_model_calls"] == spec["fresh_cases"] == 0
        and spec["automatic_retry"] is False,
        "Unchanged replay scope and budget",
    )
    review, context, boot, migrated = (read(item, refs) for item in (REVIEW, CONTEXT, BOOTSTRAP, MIGRATED_MANIFEST))
    require(
        review["complete"] is review["passed"] is context["complete"] is context["passed"] is True,
        "Closed original development review and endpoint context",
    )
    require(
        boot["complete"] is boot["passed"] is boot["bootstrap_verified"] is boot["source_unchanged"] is True,
        "Closed current CPU bootstrap",
    )
    model = boot["bootstrap"]["model_records"][2]
    require(model["stage"] == "record_xml" and model["completed"] is True, "Previously mapped record XML")
    mapped_ref = {"path": model["xml_ref"]["path"], "sha256": model["xml_ref"]["sha256"]}
    mapped_xml = checked(mapped_ref, refs).read_bytes()
    assets = {}
    for stage in boot["bootstrap"]["model_records"]:
        require(stage["completed"] is True, "Previously completed bootstrap model")
        for item in stage["assets"]:
            raw = item["reference"]
            asset_ref = {"path": raw["path"], "sha256": raw["sha256"]}
            checked(asset_ref, refs)
            require(
                asset_ref["path"] not in assets or assets[asset_ref["path"]] == asset_ref["sha256"],
                "Asset identity conflict",
            )
            assets[asset_ref["path"]] = asset_ref["sha256"]
    require(len(assets) == 96, "Original 96 unique runtime assets")
    source_map = review["source_and_artifact_sha256"]

    def reviewed(path):
        ref = reference(path)
        require(source_map.get(ref["path"]) == ref["sha256"], "Original independently reviewed input")
        return checked(ref, refs)

    cases = []
    template_xml = None
    for original_spec in expected:
        case_spec = select_prefix(original_spec, prefix_steps)
        index, n = case_spec["case_index"], case_spec["prefix_steps"]
        item = review["cases"][index]
        require(item["case"]["case_id"] == case_spec["case_id"], "Registered case identity")
        context_ref = context["case_refs"][index]
        cause = read(context_ref, refs)
        require(
            cause["last_at_or_above_buffer_endpoint"] == original_spec["prefix_steps"],
            "Registered historical diagnostic endpoint",
        )
        case_dir = (
            REPO / "outputs/tracking_preview_wam_development_v4_20261027/development/cases" / case_spec["case_id"]
        )
        worker = json.loads(reviewed(case_dir / "worker_result.json").read_text())
        require(
            worker["complete"] is worker["passed"] is True and worker["physics_steps"] == 600, "Closed original worker"
        )
        folder = Path(worker["physical_case_dir"])
        require(
            folder.is_relative_to(case_dir) and str(folder) == item["mechanical"]["case"], "Original physical directory"
        )
        manifest = json.loads(reviewed(folder / "manifest.json").read_text())
        controller = json.loads(reviewed(folder / "controller.json").read_text())
        rows = read_recorded_prefix(reviewed(folder / "steps.jsonl"), n, 600)
        endpoint_receipts = read_recorded_prefix(reviewed(Path(worker["endpoint_receipts_ref"]["path"])), n + 1, 601)
        arrays_path = reviewed(Path(worker["arrays_ref"]["path"]))
        require(
            manifest["runtime_config"]["seed"] == 43
            and manifest["record"]["task_id"] == manifest["runtime_config"]["task_id"],
            "Original task and seed",
        )
        original_xml = manifest["record"]["model_xml"].encode()
        if template_xml is None:
            template_xml = original_xml
            require(
                hashlib.sha256(template_xml).hexdigest() == boot["bootstrap"]["original_record_xml_sha256"],
                "The mapping template is the verified task0 original record",
            )
        case_xml = map_record_xml(original_xml, template_xml, mapped_xml).decode()
        commands = np.asarray([row["command"] for row in rows], dtype=float)
        with np.load(arrays_path, allow_pickle=False) as archive:
            saved = saved_prefix(archive, n)
        require(commands.shape == (n, 7) and np.isfinite(commands).all(), "Complete causal saved commands")
        require(
            hashlib.sha256(np.asarray(saved["sim_states"][0], dtype=np.float64).tobytes()).hexdigest()
            == manifest["record"]["initial_state_sha256"],
            "Original initial state SHA",
        )
        derived = dict(manifest["runtime_config"])
        for key in ("libero_root", "libero_config", "osmesa_library"):
            derived[key] = migrated["runtime_config"][key]
            require(
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
                "receipts": endpoint_receipts[: n + 1],
                "controller": controller,
            }
        )
    return {
        "spec_ref": spec_ref,
        "spec": spec,
        "cases": cases,
        "asset_refs": assets,
        "source_sha256": refs,
        "cv2_admission": boot["cv2_admission"],
    }


def source_refs():
    return [reference(p) for p in (Path(__file__), CONFIG, PLAN, CORE, RUNTIME, TEST)]


def worker(index):
    reg = read(reference(OUTPUT / "registration.json"))
    inputs = load_inputs()
    require(
        reg["source_sha256"] == inputs["source_sha256"] and reg["asset_refs"] == inputs["asset_refs"],
        "Worker inputs match the frozen owner registration",
    )
    require(reg["source_refs"] == source_refs(), "Worker source fingerprint")
    case = next(c for c in inputs["cases"] if c["spec"]["case_index"] == index)
    work = OUTPUT / f"case_{index:02d}"
    result = {"complete": False, "passed": False, "case": case["spec"], "automatic_retry": False}
    try:
        guard = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(guard["cv2_probe_ref"], guard["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime

        def progress(receipt):
            print(
                json.dumps(
                    {
                        "case_index": index,
                        "checked_endpoints": receipt.get("checked_endpoints", 0),
                        "control_attempted": receipt.get("control_steps_attempted", 0),
                        "control_completed": receipt.get("control_steps_completed", 0),
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


def owner():
    require(not OUTPUT.exists(), "New output only; no overwrite or automatic retry")
    dry_ref, qa_ref = reference(DRY), reference(QA)
    dry, qa = read(dry_ref), read(qa_ref)
    require(dry["complete"] is dry["passed"] is True and dry["physics_steps"] == 0, "Completed zero-step input dry")
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "CPU QA and worker lifecycle closed",
    )
    sources = source_refs()
    qa_sources = read(qa["source_before_ref"])["sources"]
    require(
        all(qa_sources.get(str(Path(r["path"]).relative_to(REPO))) == r["sha256"] for r in sources),
        "QA source identity",
    )
    inputs = load_inputs()
    require(
        dry["source_sha256"] == inputs["source_sha256"]
        and dry["source_refs"] == sources
        and dry["asset_refs"] == inputs["asset_refs"],
        "Dry input and source identity",
    )
    reg = publish(
        OUTPUT / "registration.json",
        {
            "dry_ref": dry_ref,
            "qa_ref": qa_ref,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "cases": [c["spec"] for c in inputs["cases"]],
            "max_control_steps": 213,
            "max_native_steps": 5325,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
    )

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Owner received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    workers, launch_error = [], None
    try:
        for index in (2, 6):
            directory = OUTPUT / f"case_{index:02d}"
            directory.mkdir()
            log = (directory / "worker.log").open("x")
            command = [
                sys.executable,
                "-m",
                "scripts.simulation.runtime.replay_multicase_development_prefix",
                "--worker",
                str(index),
            ]
            try:
                process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            workers.append({"index": index, "process": process, "log": log})
    except BaseException:
        launch_error = traceback.format_exc()
        for item in workers:
            if item["process"].poll() is None:
                try:
                    item["process"].terminate()
                    item["signaled"] = True
                except ProcessLookupError:
                    pass
    deadline = time.monotonic() + inputs["spec"]["worker_timeout_seconds"]
    lifecycle = []
    for item in workers:
        process = item["process"]
        signaled = item.get("signaled", False)
        try:
            code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            signaled = True
            process.terminate()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
        except BaseException:
            launch_error = traceback.format_exc()
            for other in workers:
                if other["process"].poll() is None:
                    try:
                        other["process"].terminate()
                        other["signaled"] = True
                    except ProcessLookupError:
                        pass
            signaled = item.get("signaled", False)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
        item["log"].close()
        lifecycle.append(
            {
                "case_index": item["index"],
                "pid": process.pid,
                "returncode": code,
                "actual_wait": True,
                "reaped": True,
                "signal_sent": signaled,
                "log_ref": reference(OUTPUT / f"case_{item['index']:02d}/worker.log"),
            }
        )
    life_ref = publish(OUTPUT / "workers.json", {"workers": lifecycle, "launch_error": launch_error})
    result = {
        "complete": False,
        "passed": False,
        "registration_ref": reg,
        "workers_ref": life_ref,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        require(
            not launch_error
            and len(lifecycle) == 2
            and all(w["returncode"] == 0 and not w["signal_sent"] for w in lifecycle),
            "Both registered workers must complete",
        )
        case_refs = [reference(OUTPUT / f"case_{i:02d}/result.json") for i in (2, 6)]
        cases = [read(ref) for ref in case_refs]
        require(all(c["complete"] is c["passed"] is True for c in cases), "Both full prefixes verified")
        require(
            all(reference(path)["sha256"] == digest for path, digest in inputs["source_sha256"].items()),
            "Inputs unchanged",
        )
        require(source_refs() == sources, "Replay source unchanged")
        result.update(
            complete=True,
            passed=True,
            case_refs=case_refs,
            sources_unchanged=True,
            control_steps=sum(c["runtime"]["control_steps_completed"] for c in cases),
            native_steps=sum(c["runtime"]["counters"]["mj_step"]["completed"] for c in cases),
        )
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", type=int, choices=(2, 6))
    args = parser.parse_args()
    environment()
    if args.dry_run:
        require(not DRY.exists(), "New dry result only")
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
                    "cases": [c["spec"] for c in inputs["cases"]],
                    "physics_steps": 0,
                    "world_model_calls": 0,
                    "fresh_cases": 0,
                },
            ),
            flush=True,
        )
    elif args.execute:
        owner()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
