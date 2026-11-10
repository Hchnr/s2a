"""Run and replay causal preview tracking from the first fixed warning."""

import argparse
import json
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.limits.causal_warning import joint_warnings
from s2a.research.native_runtime.development_continuation import current_robot_dynamics, run_continuation
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import compare_multicase_emergency_retreat as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/early_warning_tracking_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/early_warning_tracking_20261110.md"
POLICY = REPO / "src/s2a/research/joint_control/limits/warning_tracking.py"
POLICY_TEST = REPO / "tests/research/joint_control/limits/test_warning_tracking.py"
INPUT_TEST = REPO / "tests/research/native_runtime/test_early_warning_tracking_inputs.py"
WARNING = REPO / "src/s2a/research/joint_control/limits/causal_warning.py"
QA = REPO / "outputs/qa/early_warning_tracking_20261110/result.json"
DRY = REPO / "outputs/early_warning_tracking_dry_20261110/result.json"
ROOT = REPO / "outputs/early_warning_tracking_20261110"
AUDIT = {
    "path": str(REPO / "outputs/causal_joint_warning_audit_20261110/result.json"),
    "sha256": "3814fa596db7f5c76fcdacad9869468d29b6cf40776b23afe18270553ec7cad1",
}
POLICY_MODULE = "s2a.research.joint_control.limits.warning_tracking"
PREVIEW_MODULE = "s2a.research.joint_control.preview.dependency_safe_preview_runtime"
BRANCHES = (
    (2, "history"),
    (2, "preview"),
    (2, "guarded_preview"),
    (6, "history"),
    (6, "preview"),
    (6, "guarded_preview"),
)


def source_refs():
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        POLICY,
        POLICY_TEST,
        INPUT_TEST,
        WARNING,
        Path(prefix.__file__),
        prefix.CORE,
        prefix.RUNTIME,
        prefix.TEST,
        previous.CORE,
        previous.CORE_TEST,
        Path(previous.__file__),
        REPO / "src/s2a/research/joint_control/preview/dependency_safe_preview_runtime.py",
    )
    return [prefix.reference(path) for path in paths]


def first_warning_from_saved(case):
    """Recompute the first threshold crossing using only the bounded prefix."""
    xml = ET.fromstring(case["record_xml"])
    joints = list(xml.find("worldbody").iter("joint"))
    prefix.require(
        [j.get("name") for j in joints[:7]] == [f"robot0_joint{i}" for i in range(1, 8)], "First seven joint addresses"
    )
    types = [j.get("type", "hinge") for j in joints]
    prefix.require(all(t in ("hinge", "slide", "free") for t in types), "Known joint widths")
    nq = sum(7 if t == "free" else 1 for t in types)
    limits = np.asarray([[float(x) for x in j.get("range").split()] for j in joints[:7]])
    states = case["saved"]["sim_states"]
    n = case["spec"]["prefix_steps"]
    prefix.require(len(states) == n + 1, "Only past states plus current endpoint")
    alarms = [joint_warnings(s[1:8], s[1 + nq : 1 + nq + 7], limits)["static_0p10"]["alarm"] for s in states]
    prefix.require(not any(alarms[:-1]) and alarms[-1], "First causal warning must be this endpoint")
    margins = np.minimum(states[-1, 1:8] - limits[:, 0], limits[:, 1] - states[-1, 1:8])
    prefix.require(float(margins.min()) >= 0.05, "Original buffer still holds at the selected endpoint")
    return {"first_warning_endpoint": n, "minimum_margin_rad": float(margins.min()), "nq": nq}


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    expected = [
        {
            "case_index": 2,
            "case_id": "task0_init0_s43_raw",
            "prefix_steps": 43,
            "chunk_index": 5,
            "cursor": 3,
            "target": "akita_black_bowl_1",
        },
        {
            "case_index": 6,
            "case_id": "task1_init0_s43_raw",
            "prefix_steps": 52,
            "chunk_index": 6,
            "cursor": 4,
            "target": "wine_bottle_1",
        },
    ]
    prefix.require(
        config["schema"] == "s2a.early-warning-tracking.v1" and config["cases"] == expected,
        "Fixed two first-warning cases",
    )
    prefix.require(
        config["branches"] == ["history", "preview", "guarded_preview"]
        and config["continuation_steps"] == 8
        and config["joint_warning_margin_rad"] == 0.1
        and config["joint_retreat_speed_rad_s"] == 0.3
        and config["control_dt_s"] == 0.05
        and config["arm_rate_limit"] == 0.25
        and config["position_cap_m"] == 0.01
        and config["rotation_cap_rad"] == 0.05
        and config["joint_buffer_rad"] == 0.05
        and config["absolute_comparison_tolerance"] == 1e-10
        and config["relative_comparison_tolerance"] == 0
        and config["max_control_steps_per_phase"] == 333
        and config["max_native_steps_per_phase"] == 8325
        and config["max_models_per_phase"] == config["max_contexts_per_phase"] == 18
        and config["max_preview_calls_per_phase"] == 32
        and config["worker_timeout_seconds"] == 1800
        and config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "Original experiment gates and budgets",
    )
    audit = prefix.read(AUDIT)
    prefix.require(
        audit["complete"] is audit["passed"] is audit["sources_unchanged"] is True, "Closed twelve-case warning audit"
    )
    original = prefix.load_inputs({2: 43, 6: 52})
    review = prefix.read(prefix.REVIEW)
    source_map = review["source_and_artifact_sha256"]
    refs = dict(original["source_sha256"])
    refs[str(CONFIG)] = config_ref["sha256"]
    refs[AUDIT["path"]] = AUDIT["sha256"]
    cases = []
    for spec in expected:
        case = next(item for item in original["cases"] if item["spec"]["case_index"] == spec["case_index"])
        first = first_warning_from_saved(case)
        folder = Path(review["cases"][spec["case_index"]]["mechanical"]["case"])
        prefix.require(folder.is_relative_to(REPO / "outputs"), "Reviewed local development directory")
        chunk_path, tool_path = folder / "chunks.json", Path(case["config"]["tool_contract"])
        for path in (chunk_path, tool_path):
            ref = prefix.reference(path)
            prefix.require(source_map.get(ref["path"]) == ref["sha256"], "Reviewed chunk/tool identity")
            refs[ref["path"]] = ref["sha256"]
        chunk = previous.read_array_item(chunk_path, spec["chunk_index"])
        prefix.require(
            chunk["step"] == spec["chunk_index"] * 8 and chunk["step"] + spec["cursor"] == spec["prefix_steps"],
            "Chunk already generated",
        )
        targets = np.asarray(chunk["effective_targets_tcp"], dtype=float)
        anchor = np.asarray(chunk["anchor_tcp"], dtype=float)
        tool = np.asarray(json.loads(tool_path.read_text())["source_from_tcp"], dtype=float)
        prefix.require(
            targets.shape == (40, 4, 4)
            and anchor.shape == tool.shape == (4, 4)
            and np.isfinite(targets).all()
            and np.isfinite(anchor).all()
            and np.isfinite(tool).all(),
            "Finite generated reference and transforms",
        )
        case.update(
            intervention_spec=spec,
            physical_folder=folder,
            targets=targets,
            anchor=anchor,
            fixed_targets=targets[spec["cursor"] : spec["cursor"] + 8].copy(),
            tool=tool,
            first_warning=first,
        )
        cases.append(case)
    return {
        "config": config,
        "config_ref": config_ref,
        "cases": cases,
        "asset_refs": original["asset_refs"],
        "source_sha256": refs,
        "cv2_admission": original["cv2_admission"],
    }


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "Fixed registered branch")
    return f"case_{index:02d}_{mode}"


def current_dynamics(env, jacobian):
    """Read actual mass, joint velocity and site from the current simulator state."""
    return current_robot_dynamics(env, jacobian)


def make_action(phase, mode, case, env, receipt, expected_periods):
    if phase == "checker":
        prefix.require(
            POLICY_MODULE not in sys.modules and PREVIEW_MODULE not in sys.modules,
            "Checker must not import preview or guard",
        )
        return previous.make_action(mode, case, expected_periods=expected_periods)
    if mode == "history":
        return previous.make_action("history", case)
    from s2a.research.joint_control.limits.warning_tracking import select_tracking_action, site_references

    poses, anchor = site_references(
        case["targets"], case["anchor"], case["tool"], case["controller"]["source_from_site"]
    )
    cursor = case["intervention_spec"]["cursor"]
    contract = case["controller"]["contract"]

    def action(q, limits, jacobian, previous_command, offset):
        mass, velocity, site = current_dynamics(env, jacobian)
        receipt["preview_calls"]["attempted"] += 1
        prefix.require(receipt["preview_calls"]["attempted"] <= 8, "Bounded branch preview calls")
        selected = select_tracking_action(
            mass,
            jacobian,
            q,
            velocity,
            limits,
            site,
            poses,
            anchor,
            cursor + offset,
            previous_command[:6],
            contract,
            guard=mode == "guarded_preview",
        )
        receipt["preview_calls"]["completed"] += 1
        selected["command"] = np.r_[selected["command"], 0.0]
        selected["newly_selected"] = True
        return selected

    return action, None


def worker(phase, index, mode):
    def interrupted(signum, _frame):
        raise InterruptedError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    directory = ROOT / phase / branch_key(index, mode)
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        inputs["source_sha256"] == registration["source_sha256"]
        and inputs["asset_refs"] == registration["asset_refs"]
        and source_refs() == registration["source_refs"],
        "Frozen input and source identity",
    )
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
    expected, expected_stop = None, None
    if phase == "checker":
        original = prefix.read(prefix.reference(ROOT / "producer" / branch_key(index, mode) / "result.json"))
        prefix.require(original["complete"] is original["passed"] is True, "Closed producer branch")
        expected = original["runtime"]["continuation"]["periods"]
        expected_stop = original["runtime"]["continuation"]["stop"]
    result = {
        "complete": False,
        "passed": False,
        "phase": phase,
        "case_index": index,
        "branch": mode,
        "automatic_retry": False,
    }
    try:
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            receipt["preview_calls"] = {"attempted": 0, "completed": 0}
            action, historical = make_action(phase, mode, case, env, receipt, expected)
            return run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=case["spec"]["prefix_steps"],
                previous_command=case["commands"][-1],
                targets=case["fixed_targets"],
                tool=case["tool"],
                target_object=case["intervention_spec"]["target"],
                choose_action=action,
                stop_on_outside=mode != "history",
                expected_periods=expected,
                historical_states=historical,
                progress=progress,
            )

        def progress(receipt):
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "case": index,
                        "branch": mode,
                        "checked_prefix_endpoints": receipt.get("checked_endpoints", 0),
                        "control_completed": receipt.get("control_steps_completed", 0),
                        "new_commands": receipt.get("newly_selected_commands", 0),
                        "preview_calls": receipt.get("preview_calls"),
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
                directory / "runtime",
                inputs["asset_refs"],
                case["controller"],
                continuation=continuation,
                continuation_budget=8,
                progress=progress,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        if phase == "checker":
            prefix.require(
                POLICY_MODULE not in sys.modules and PREVIEW_MODULE not in sys.modules,
                "Independent checker never imported decision modules",
            )
            prefix.require(
                receipt["continuation"]["stop"] == expected_stop
                and receipt["newly_selected_commands"] == 0
                and receipt["preview_calls"]["attempted"] == 0,
                "Original stop and no checker decisions",
            )
        result.update(
            complete=True,
            passed=True,
            runtime=receipt,
            candidate_policy_imported=POLICY_MODULE in sys.modules,
            preview_module_imported=PREVIEW_MODULE in sys.modules,
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def run_phase(phase):
    directory = ROOT / phase
    prefix.require(not directory.exists(), "New phase output only; no retry")
    dry_ref, qa_ref = prefix.reference(DRY), prefix.reference(QA)
    dry, qa = prefix.read(dry_ref), prefix.read(qa_ref)
    prefix.require(dry["complete"] is dry["passed"] is True and dry["physics_steps"] == 0, "Dry admission")
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "CPU QA admission",
    )
    source_before = snapshot(REPO)["sources"]
    prefix.require(source_before == prefix.read(qa["source_before_ref"])["sources"], "Full QA source identity")
    sources, inputs = source_refs(), load_inputs()
    prefix.require(
        dry["source_refs"] == sources
        and dry["source_sha256"] == inputs["source_sha256"]
        and dry["asset_refs"] == inputs["asset_refs"],
        "Dry source and input identity",
    )
    producer_ref = None
    if phase == "checker":
        producer_ref = prefix.reference(ROOT / "producer/result.json")
        producer = prefix.read(producer_ref)
        prefix.require(
            producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
            and len(producer["branch_refs"]) == 6,
            "Closed producer before checker",
        )
        for ref in producer["branch_refs"]:
            prefix.checked(ref)
    registration = publish(
        directory / "registration.json",
        {
            "dry_ref": dry_ref,
            "qa_ref": qa_ref,
            "audit_ref": AUDIT,
            "producer_ref": producer_ref,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "branches": [list(item) for item in BRANCHES],
            "max_control_steps": 333,
            "max_native_steps": 8325,
            "max_preview_calls": 32,
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
        for index, mode in BRANCHES:
            path = directory / branch_key(index, mode)
            path.mkdir()
            log = (path / "worker.log").open("x")
            command = [
                sys.executable,
                "-m",
                "scripts.simulation.runtime.compare_early_warning_tracking",
                "--worker",
                phase,
                str(index),
                mode,
            ]
            try:
                proc = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            workers.append({"index": index, "mode": mode, "process": proc, "log": log})
    except BaseException:
        launch_error = traceback.format_exc()
        for item in workers:
            if item["process"].poll() is None:
                try:
                    item["process"].terminate()
                    item["signaled"] = True
                except ProcessLookupError:
                    pass
    deadline = time.monotonic() + inputs["config"]["worker_timeout_seconds"]
    lifecycle = []
    for item in workers:
        proc, sent = item["process"], item.get("signaled", False)
        try:
            code = proc.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            sent = True
            proc.terminate()
            try:
                code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait()
        except BaseException:
            launch_error = traceback.format_exc()
            for other in workers:
                if other["process"].poll() is None:
                    try:
                        other["process"].terminate()
                        other["signaled"] = True
                    except ProcessLookupError:
                        pass
            sent = item.get("signaled", False)
            try:
                code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait()
        item["log"].close()
        lifecycle.append(
            {
                "case_index": item["index"],
                "branch": item["mode"],
                "pid": proc.pid,
                "returncode": code,
                "actual_wait": True,
                "reaped": True,
                "signal_sent": sent,
                "log_ref": prefix.reference(directory / branch_key(item["index"], item["mode"]) / "worker.log"),
            }
        )
    workers_ref = publish(directory / "workers.json", {"workers": lifecycle, "launch_error": launch_error})
    result = {
        "complete": False,
        "passed": False,
        "phase": phase,
        "registration_ref": registration,
        "workers_ref": workers_ref,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        prefix.require(
            not launch_error
            and len(lifecycle) == 6
            and all(w["returncode"] == 0 and not w["signal_sent"] for w in lifecycle),
            "All workers closed",
        )
        refs = [prefix.reference(directory / branch_key(i, mode) / "result.json") for i, mode in BRANCHES]
        branches = [prefix.read(ref) for ref in refs]
        prefix.require(all(b["complete"] is b["passed"] is True for b in branches), "Complete branch results")
        prefix.require(
            all(prefix.reference(path)["sha256"] == digest for path, digest in inputs["source_sha256"].items())
            and source_refs() == sources
            and snapshot(REPO)["sources"] == source_before,
            "Inputs and full sources unchanged",
        )
        controls = sum(b["runtime"]["control_steps_completed"] for b in branches)
        native = sum(b["runtime"]["counters"]["mj_step"]["completed"] for b in branches)
        models = sum(b["runtime"]["counters"]["model_construct"]["completed"] for b in branches)
        contexts = sum(b["runtime"]["counters"]["context"]["completed"] for b in branches)
        previews = sum(b["runtime"]["preview_calls"]["attempted"] for b in branches)
        prefix.require(
            controls <= 333
            and native <= 8325
            and native == controls * 25
            and models <= 18
            and contexts <= 18
            and previews <= 32,
            "Original phase budgets",
        )
        if phase == "checker":
            prefix.require(
                previews == 0
                and all(
                    not b["candidate_policy_imported"]
                    and not b["preview_module_imported"]
                    and b["runtime"]["newly_selected_commands"] == 0
                    for b in branches
                ),
                "Checker consumed saved actions without decisions",
            )
        result.update(
            complete=True,
            passed=True,
            branch_refs=refs,
            sources_unchanged=True,
            control_steps=controls,
            native_steps=native,
            model_initializations=models,
            software_contexts=contexts,
            preview_calls=previews,
            newly_selected_commands=sum(b["runtime"]["newly_selected_commands"] for b in branches),
        )
    finally:
        print(publish(directory / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--produce", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--worker", nargs=3, metavar=("PHASE", "CASE", "BRANCH"))
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        prefix.require(not DRY.exists(), "New dry output only")
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
                    "cases": [x["intervention_spec"] for x in inputs["cases"]],
                    "first_warnings": [x["first_warning"] for x in inputs["cases"]],
                    "physics_steps": 0,
                    "world_model_calls": 0,
                    "fresh_cases": 0,
                },
            ),
            flush=True,
        )
    elif args.produce:
        run_phase("producer")
    elif args.check:
        run_phase("checker")
    else:
        phase, index, mode = args.worker
        prefix.require(phase in ("producer", "checker"), "Registered phase")
        worker(phase, int(index), mode)


if __name__ == "__main__":
    main()
