"""Qualify task-6 saved-prefix replay at the first two causal warnings."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.limits.causal_warning import joint_warnings
from s2a.research.native_runtime.development_prefix_replay import map_record_xml, saved_prefix
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_causal_event_prefixes as baseline
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_warning_prefix_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_warning_prefix_20261111.md"
QA = REPO / "outputs/qa/task2_warning_prefix_20261111/result.json"
DRY = REPO / "outputs/task2_warning_prefix_dry_20261111/result.json"
OUTPUT = REPO / "outputs/task2_warning_prefix_20261111"
WARNINGS = {
    "path": str(REPO / "outputs/causal_joint_warning_audit_20261110/result.json"),
    "sha256": "3814fa596db7f5c76fcdacad9869468d29b6cf40776b23afe18270553ec7cad1",
}
CASE_BASE = REPO / "outputs/tracking_preview_wam_development_v4_20261027/development/cases"
CASES = [
    dict(case_index=10, case_id="task2_init0_s43_raw", prefix_steps=34),
    dict(case_index=11, case_id="task2_init0_s43_terminal058", prefix_steps=82),
]


def source_refs():
    return baseline.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/native_runtime/task2_prefix_binding.py",
            REPO / "src/s2a/research/joint_control/limits/causal_warning.py",
            REPO / "tests/research/native_runtime/test_task2_prefix_binding.py",
            REPO / "tests/research/native_runtime/test_task2_warning_prefix.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == {
            "schema": "s2a.task2-warning-prefix.v1",
            "cases": CASES,
            "task_id": 6,
            "seed": 43,
            "warning_rule": "velocity_0p2s",
            "joint_buffer_rad": 0.05,
            "control_dt_s": 0.05,
            "inner_dt_s": 0.002,
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 0,
            "max_control_steps": 116,
            "max_inner_steps": 2900,
            "max_models": 6,
            "max_contexts": 6,
            "max_workers": 2,
            "worker_timeout_seconds": 1800,
            "new_actions": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Fixed third-task warning-prefix scope and budget",
    )
    return ref, config


def check_warning(spec, warning_case, saved):
    """Recompute the registered warning from only the current q and qvel."""
    n = spec["prefix_steps"]
    prefix.require(
        warning_case["case"]["case_index"] == spec["case_index"]
        and warning_case["case"]["case_id"] == spec["case_id"]
        and warning_case["rules"]["velocity_0p2s"]["first_alarm_while_buffer_safe_before_first_outside"] == n,
        "Original audited first safe warning and case identity",
    )
    state = np.asarray(saved["sim_states"][-1], dtype=float)
    nq = warning_case["nq"]
    prefix.require(
        type(nq) is int and nq >= 7 and state.ndim == 1 and len(state) >= 1 + nq + 7, "Original state layout"
    )
    q, velocity = state[1:8], state[1 + nq : 1 + nq + 7]
    limits = np.asarray(warning_case["joint_limits_rad"], dtype=float)
    current = joint_warnings(q, velocity, limits)["velocity_0p2s"]
    recorded = warning_case["observations"][n]["warnings"]["velocity_0p2s"]
    for key in ("alarm", "joint_index", "side"):
        prefix.require(current[key] == recorded[key], f"Original warning {key} differs")
    for key in ("margin_rad", "outward_velocity_rad_s", "projected_margin_rad", "threshold_rad"):
        np.testing.assert_allclose(current[key], recorded[key], atol=1e-10, rtol=0)
    margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
    prefix.require(current["alarm"] and margin >= 0.05, "Warning is causal and current endpoint retains buffer")
    return {"endpoint": n, "minimum_margin_rad": margin, "warning": current}


def load_inputs():
    config_ref, config = read_config()
    refs = {config_ref["path"]: config_ref["sha256"]}
    review, warnings, migrated = (prefix.read(ref, refs) for ref in (prefix.REVIEW, WARNINGS, prefix.MIGRATED_MANIFEST))
    prior = baseline.load_inputs()
    for path, digest in prior["source_sha256"].items():
        prefix.require(path not in refs or refs[path] == digest, "Original source identity conflict")
        refs[path] = digest
    prefix.require(
        review["complete"]
        is review["passed"]
        is warnings["complete"]
        is warnings["passed"]
        is warnings["sources_unchanged"]
        is True
        and len(warnings["case_refs"]) == 12,
        "Original twelve-case warning audit and review closed",
    )
    source_map = review["source_and_artifact_sha256"]

    def reviewed(path):
        ref = prefix.reference(path)
        prefix.require(source_map.get(ref["path"]) == ref["sha256"], "Original reviewed input identity")
        return prefix.checked(ref, refs)

    template = prior["cases"][0]
    template_original = template["record"]["model_xml"].encode()
    template_mapped = template["record_xml"].encode()
    cases = []
    for spec in CASES:
        index, n = spec["case_index"], spec["prefix_steps"]
        original = review["cases"][index]
        case = original["case"]
        prefix.require(
            case["case_id"] == spec["case_id"]
            and case["case_index"] == index
            and case["sampling_seed"] == config["seed"]
            and original["mechanical"]["steps"] == 600
            and original["mechanical"]["chunks"] == 75,
            "Original third-task development case",
        )
        warning_case = prefix.read(warnings["case_refs"][index], refs)
        case_dir = CASE_BASE / spec["case_id"]
        worker = json.loads(reviewed(case_dir / "worker_result.json").read_text())
        folder = Path(worker["physical_case_dir"])
        prefix.require(
            worker["complete"] is worker["passed"] is True
            and worker["physics_steps"] == 600
            and folder.is_relative_to(case_dir)
            and str(folder) == original["mechanical"]["case"],
            "Original completed development worker and case folder",
        )
        manifest = json.loads(reviewed(folder / "manifest.json").read_text())
        controller = json.loads(reviewed(folder / "controller.json").read_text())
        rows = prefix.read_recorded_prefix(reviewed(folder / "steps.jsonl"), n, 600)
        receipts = prefix.read_recorded_prefix(reviewed(Path(worker["endpoint_receipts_ref"]["path"])), n + 1, 601)
        with np.load(reviewed(Path(worker["arrays_ref"]["path"])), allow_pickle=False) as archive:
            saved = saved_prefix(archive, n)
        prefix.require(
            manifest["runtime_config"]["task_id"] == manifest["record"]["task_id"] == config["task_id"]
            and manifest["runtime_config"]["seed"] == config["seed"]
            and hashlib.sha256(np.asarray(saved["sim_states"][0], dtype=np.float64).tobytes()).hexdigest()
            == manifest["record"]["initial_state_sha256"],
            "Exact original third-task initial state, task and seed",
        )
        xml = map_record_xml(manifest["record"]["model_xml"].encode(), template_original, template_mapped).decode()
        runtime = dict(manifest["runtime_config"])
        for key in ("libero_root", "libero_config", "osmesa_library"):
            runtime[key] = migrated["runtime_config"][key]
            prefix.require(
                Path(runtime[key]).is_absolute() and Path(runtime[key]).is_relative_to(REPO),
                "Migrated repository runtime",
            )
        commands = np.asarray([row["command"] for row in rows], dtype=float)
        prefix.require(commands.shape == (n, 7) and np.isfinite(commands).all(), "Only actual completed command prefix")
        warning = check_warning(spec, warning_case, saved)
        cases.append(
            dict(
                spec=spec,
                config=runtime,
                record=manifest["record"],
                record_xml=xml,
                commands=commands,
                saved=saved,
                receipts=receipts,
                controller=controller,
                warning=warning,
            )
        )
    return dict(
        config_ref=config_ref,
        config=config,
        cases=cases,
        asset_refs=prior["asset_refs"],
        source_sha256=refs,
        cv2_admission=prior["cv2_admission"],
    )


def input_receipt(inputs):
    return dict(
        config_ref=inputs["config_ref"],
        cases=[case["spec"] for case in inputs["cases"]],
        asset_refs=inputs["asset_refs"],
        source_sha256=inputs["source_sha256"],
    )


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen replay input and source identity",
    )
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        case=case["spec"],
        warning=case["warning"],
        automatic_retry=False,
    )
    try:
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.task2_prefix_binding import IsolatedTask2PrefixRuntime

        def progress(receipt):
            completed = receipt.get("control_steps_completed", 0)
            if completed % 16 == 0 or completed == case["spec"]["prefix_steps"]:
                print(
                    json.dumps(
                        dict(
                            case_index=index, control_completed=completed, native_steps=receipt["counters"]["mj_step"]
                        ),
                        allow_nan=False,
                    ),
                    flush=True,
                )

        try:
            runtime = IsolatedTask2PrefixRuntime().run(
                case["config"],
                case["record"],
                case["record_xml"],
                case["commands"],
                case["saved"],
                case["receipts"],
                OUTPUT / f"case_{index:02d}" / "runtime",
                inputs["asset_refs"],
                case["controller"],
                registered_seed=43,
                capture_final_prefix_inner=True,
                progress=progress,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        result.update(complete=True, passed=True, runtime=runtime)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / f"case_{index:02d}" / "result.json", result), flush=True)


def execute():
    prefix.environment()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Frozen CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "QA and replay source identity")
    inputs, sources = load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry identity",
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration = publish(OUTPUT / "registration.json", dict(inputs=input_receipt(inputs), source_refs=sources))
    specs = [
        dict(
            key=f"case_{item['case_index']:02d}",
            argv=[
                sys.executable,
                "-m",
                "scripts.simulation.runtime.replay_task2_warning_prefix",
                "--worker",
                str(item["case_index"]),
            ],
            log=OUTPUT / f"case_{item['case_index']:02d}" / "worker.log",
        )
        for item in CASES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers = publish(OUTPUT / "workers.json", lifecycle)
    available = [
        prefix.reference(OUTPUT / f"case_{item['case_index']:02d}" / "result.json")
        for item in CASES
        if (OUTPUT / f"case_{item['case_index']:02d}" / "result.json").is_file()
    ]
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        registration_ref=registration,
        workers_ref=workers,
        available_case_refs=available,
        automatic_retry=False,
        world_model_calls=0,
        fresh_cases=0,
    )
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 2
            and all(
                w["returncode"] == 0 and w["actual_wait"] and w["reaped"] and not w["signal_sent"]
                for w in lifecycle["workers"]
            ),
            "Both task2 prefix workers exited and were reaped",
        )
        case_refs = [prefix.reference(OUTPUT / f"case_{item['case_index']:02d}" / "result.json") for item in CASES]
        cases = [prefix.read(ref) for ref in case_refs]
        prefix.require(
            all(item["complete"] is item["passed"] is item["runtime"]["cleanup_complete"] is True for item in cases),
            "Both original physical prefixes closed",
        )
        runtimes = [item["runtime"] for item in cases]
        counts = dict(
            control_steps=sum(r["control_steps_completed"] for r in runtimes),
            inner_steps=sum(r["counters"]["mj_step"]["completed"] for r in runtimes),
            model_initializations=sum(r["counters"]["model_construct"]["completed"] for r in runtimes),
            contexts=sum(r["counters"]["context"]["completed"] for r in runtimes),
            new_actions=sum(r["newly_selected_commands"] for r in runtimes),
        )
        prefix.require(
            counts == dict(control_steps=116, inner_steps=2900, model_initializations=6, contexts=6, new_actions=0),
            "Exact registered task2 replay cost",
        )
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "Source and asset identities unchanged"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, case_refs=case_refs, counts=counts)
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", type=int, choices=[10, 11])
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        inputs = load_inputs()
        print(
            publish(
                DRY,
                dict(
                    complete=True,
                    passed=True,
                    inputs=input_receipt(inputs),
                    source_refs=source_refs(),
                    model_initializations=0,
                    physics_steps=0,
                ),
            )
        )
    elif args.run:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
