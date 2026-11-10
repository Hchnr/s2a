"""Execute and independently replay bounded emergency-retreat comparisons."""

import argparse
import json
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.infrastructure.repository_paths import repository_root
from s2a.research.native_runtime.development_continuation import run_continuation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = repository_root(anchor=__file__)
CONFIG = REPO / "configs/tracking/joint_control/multicase_emergency_retreat_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/multicase_emergency_retreat_20261110.md"
POLICY = REPO / "src/s2a/research/joint_control/limits/emergency_retreat.py"
POLICY_TEST = REPO / "tests/research/joint_control/limits/test_emergency_retreat.py"
CORE = REPO / "src/s2a/research/native_runtime/development_continuation.py"
CORE_TEST = REPO / "tests/research/native_runtime/test_development_continuation.py"
PREFIX_TEST = REPO / "tests/research/native_runtime/test_development_prefix_replay.py"
INPUT_TEST = REPO / "tests/research/native_runtime/test_multicase_emergency_inputs.py"
QA = REPO / "outputs/qa/multicase_emergency_retreat_20261110/result.json"
DRY = REPO / "outputs/multicase_emergency_retreat_dry_20261110/result.json"
ROOT = REPO / "outputs/multicase_emergency_retreat_20261110"
PRIOR = {
    "path": str(REPO / "outputs/multicase_native_prefix_replay_20261110/result.json"),
    "sha256": "1269464d34161f999937d4b2d61c5e2448f1d611353b836b1d4e08a37840980b",
}
POLICY_MODULE = "s2a.research.joint_control.limits.emergency_retreat"
BRANCHES = ((2, "history"), (2, "zero"), (2, "retreat"), (6, "history"), (6, "zero"), (6, "retreat"))


def require(value, message):
    if not value:
        raise ValueError(message)


def read_array_item(path, index):
    """Decode through one selected array item, never decode subsequent queries."""
    text = Path(path).read_text()
    decoder = json.JSONDecoder()
    position = 0
    while position < len(text) and text[position].isspace():
        position += 1
    require(position < len(text) and text[position] == "[", "Reference chunk array")
    position += 1
    for item_index in range(index + 1):
        while position < len(text) and text[position].isspace():
            position += 1
        if item_index:
            require(position < len(text) and text[position] == ",", "Reference array separator")
            position += 1
            while position < len(text) and text[position].isspace():
                position += 1
        item, position = decoder.raw_decode(text, position)
    return item


def source_refs():
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        POLICY,
        POLICY_TEST,
        CORE,
        CORE_TEST,
        PREFIX_TEST,
        INPUT_TEST,
        prefix.CONFIG,
        prefix.PLAN,
        prefix.CORE,
        prefix.RUNTIME,
        Path(prefix.__file__),
    )
    return [prefix.reference(path) for path in paths]


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    expected = [
        {
            "case_index": 2,
            "case_id": "task0_init0_s43_raw",
            "prefix_steps": 114,
            "chunk_index": 14,
            "cursor": 2,
            "target": "akita_black_bowl_1",
        },
        {
            "case_index": 6,
            "case_id": "task1_init0_s43_raw",
            "prefix_steps": 99,
            "chunk_index": 12,
            "cursor": 3,
            "target": "wine_bottle_1",
        },
    ]
    require(
        config["schema"] == "s2a.multicase-emergency-retreat.v1" and config["cases"] == expected,
        "Fixed two development cases",
    )
    require(
        config["branches"] == ["history", "zero", "retreat"]
        and config["continuation_steps"] == 8
        and config["joint_retreat_speed_rad_s"] == 0.3
        and config["control_dt_s"] == 0.05
        and config["arm_rate_limit"] == 0.25
        and config["position_cap_m"] == 0.01
        and config["rotation_cap_rad"] == 0.05
        and config["joint_buffer_rad"] == 0.05
        and config["absolute_comparison_tolerance"] == 1e-10
        and config["relative_comparison_tolerance"] == 0
        and config["max_control_steps_per_phase"] == 687
        and config["max_native_steps_per_phase"] == 17175
        and config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "Original parameters, quality gates and budgets",
    )
    prior = prefix.read(PRIOR)
    require(
        prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True
        and prior["control_steps"] == 213
        and prior["native_steps"] == 5325,
        "Both prefixes previously closed",
    )
    old = prefix.load_inputs()
    review = prefix.read(prefix.REVIEW)
    source_map = review["source_and_artifact_sha256"]
    refs = dict(old["source_sha256"])
    refs[str(CONFIG)] = config_ref["sha256"]
    refs[PRIOR["path"]] = PRIOR["sha256"]
    cases = []
    for spec in expected:
        case = next(item for item in old["cases"] if item["spec"]["case_index"] == spec["case_index"])
        require(case["spec"]["prefix_steps"] == spec["prefix_steps"], "Original prefix length")
        folder = Path(review["cases"][spec["case_index"]]["mechanical"]["case"])
        require(folder.is_relative_to(REPO / "outputs"), "Current reviewed development folder")
        chunk_path = folder / "chunks.json"
        tool_path = Path(case["config"]["tool_contract"])
        for path in (chunk_path, tool_path):
            ref = prefix.reference(path)
            require(source_map.get(ref["path"]) == ref["sha256"], "Previously reviewed reference/tool input")
            refs[ref["path"]] = ref["sha256"]
        chunk = read_array_item(chunk_path, spec["chunk_index"])
        require(
            chunk["step"] == spec["chunk_index"] * 8 and chunk["step"] + spec["cursor"] == spec["prefix_steps"],
            "Reference was generated no later than the intervention endpoint",
        )
        targets = np.asarray(chunk["effective_targets_tcp"], dtype=float)
        require(targets.shape == (40, 4, 4) and np.isfinite(targets).all(), "One complete finite generated chunk")
        fixed = targets[spec["cursor"] : spec["cursor"] + 8].copy()
        tool = np.asarray(json.loads(tool_path.read_text())["source_from_tcp"], dtype=float)
        require(fixed.shape == (8, 4, 4) and tool.shape == (4, 4), "Eight fixed reference targets and tool")
        case.update(intervention_spec=spec, fixed_targets=fixed, tool=tool, physical_folder=folder)
        cases.append(case)
    return {
        "config": config,
        "config_ref": config_ref,
        "cases": cases,
        "asset_refs": old["asset_refs"],
        "source_sha256": refs,
        "cv2_admission": old["cv2_admission"],
    }


def history_inputs(case):
    """Only the explicitly retrospective history arm may decode future records."""
    n = case["intervention_spec"]["prefix_steps"]
    selected = []
    with (case["physical_folder"] / "steps.jsonl").open() as stream:
        for index, line in enumerate(stream):
            if n <= index < n + 8:
                selected.append(json.loads(line)["command"])
            if index >= n + 7:
                break
    archive = case["saved"]
    del archive
    case_dir = case["physical_folder"].parents[2]
    worker = json.loads((case_dir / "worker_result.json").read_text())
    with np.load(prefix.checked(worker["arrays_ref"]), allow_pickle=False) as arrays:
        states = arrays["sim_states"][n + 1 : n + 9].copy()
    commands = np.asarray(selected, dtype=float)
    require(commands.shape == (8, 7) and states.shape == (8, 79), "Eight retained historical actions and states")
    return commands, states


def branch_key(index, mode):
    require((index, mode) in BRANCHES, "Fixed registered branch")
    return f"case_{index:02d}_{mode}"


def make_action(mode, case, expected_periods=None):
    """Checker returns recorded actions and never imports candidate decisions."""
    if expected_periods is not None:
        require(POLICY_MODULE not in sys.modules, "Checker must not import candidate policy")

        def action(_q, _limits, _jacobian, _previous, offset):
            return {"command": np.asarray(expected_periods[offset]["command"], dtype=float), "newly_selected": False}

        return action, None
    if mode == "history":
        actions, states = history_inputs(case)

        def action(_q, _limits, _jacobian, _previous, offset):
            return {"command": actions[offset].copy(), "newly_selected": False}

        return action, states
    from s2a.research.joint_control.limits.emergency_retreat import jacobian_retreat, rate_to_zero

    contract = case["controller"]["contract"]
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))

    def action(q, limits, jacobian, previous, _offset):
        result = (
            rate_to_zero(previous[:6], interface)
            if mode == "zero"
            else jacobian_retreat(q, limits, jacobian, previous[:6], interface)
        )
        result["command"] = np.r_[result["command"], 0.0]
        result["newly_selected"] = True
        return result

    return action, None


def worker(phase, index, mode):
    def interrupted(signum, _frame):
        raise InterruptedError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    directory = ROOT / phase / branch_key(index, mode)
    reg = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = load_inputs()
    require(
        inputs["source_sha256"] == reg["source_sha256"] and inputs["asset_refs"] == reg["asset_refs"],
        "Frozen branch input identity",
    )
    require(source_refs() == reg["source_refs"], "Frozen branch source identity")
    case = next(x for x in inputs["cases"] if x["intervention_spec"]["case_index"] == index)
    expected = None
    expected_stop = None
    if phase == "checker":
        original = prefix.read(prefix.reference(ROOT / "producer" / branch_key(index, mode) / "result.json"))
        require(original["complete"] is original["passed"] is True, "Closed producer branch")
        expected = original["runtime"]["continuation"]["periods"]
        expected_stop = original["runtime"]["continuation"]["stop"]
    choose_action, historical_states = make_action(mode, case, expected_periods=expected)
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
                choose_action=choose_action,
                stop_on_outside=mode != "history",
                expected_periods=expected,
                historical_states=historical_states,
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
                        "control_attempted": receipt.get("control_steps_attempted", 0),
                        "control_completed": receipt.get("control_steps_completed", 0),
                        "new_commands": receipt.get("newly_selected_commands", 0),
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
            require(POLICY_MODULE not in sys.modules, "Independent checker never imported policy")
            require(
                receipt["continuation"]["stop"] == expected_stop and receipt["newly_selected_commands"] == 0,
                "Original stop and no new checker decisions",
            )
        result.update(
            complete=True, passed=True, runtime=receipt, candidate_policy_imported=POLICY_MODULE in sys.modules
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def run_phase(phase):
    directory = ROOT / phase
    require(not directory.exists(), "New phase output only; no overwrite or retry")
    dry_ref, qa_ref = prefix.reference(DRY), prefix.reference(QA)
    dry, qa = prefix.read(dry_ref), prefix.read(qa_ref)
    require(dry["complete"] is dry["passed"] is True and dry["physics_steps"] == 0, "Zero-step dry closed")
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "CPU QA closed",
    )
    sources = source_refs()
    qa_sources = prefix.read(qa["source_before_ref"])["sources"]
    require(
        all(qa_sources.get(str(Path(r["path"]).relative_to(REPO))) == r["sha256"] for r in sources),
        "QA source identity",
    )
    inputs = load_inputs()
    require(
        dry["source_sha256"] == inputs["source_sha256"]
        and dry["asset_refs"] == inputs["asset_refs"]
        and dry["source_refs"] == sources,
        "Dry source and input identity",
    )
    producer_ref = None
    if phase == "checker":
        producer_ref = prefix.reference(ROOT / "producer/result.json")
        producer = prefix.read(producer_ref)
        require(producer["complete"] is producer["passed"] is True, "Producer complete before independent replay")
        require(producer["sources_unchanged"] is True and len(producer["branch_refs"]) == 6, "Six closed producer arms")
        for ref in producer["branch_refs"]:
            prefix.checked(ref)
    reg = publish(
        directory / "registration.json",
        {
            "dry_ref": dry_ref,
            "qa_ref": qa_ref,
            "prior_ref": PRIOR,
            "producer_ref": producer_ref,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "branches": [list(x) for x in BRANCHES],
            "max_control_steps": 687,
            "max_native_steps": 17175,
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
                "scripts.simulation.runtime.compare_multicase_emergency_retreat",
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
        proc = item["process"]
        sent = item.get("signaled", False)
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
        "registration_ref": reg,
        "workers_ref": workers_ref,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        require(
            not launch_error
            and len(lifecycle) == 6
            and all(w["returncode"] == 0 and not w["signal_sent"] for w in lifecycle),
            "All six branches must close",
        )
        branch_refs = [prefix.reference(directory / branch_key(i, mode) / "result.json") for i, mode in BRANCHES]
        branches = [prefix.read(ref) for ref in branch_refs]
        require(all(b["complete"] is b["passed"] is True for b in branches), "Complete branch results")
        require(
            all(prefix.reference(path)["sha256"] == digest for path, digest in inputs["source_sha256"].items()),
            "Registered inputs unchanged",
        )
        require(source_refs() == sources, "Registered sources unchanged")
        controls = sum(b["runtime"]["control_steps_completed"] for b in branches)
        native = sum(b["runtime"]["counters"]["mj_step"]["completed"] for b in branches)
        models = sum(b["runtime"]["counters"]["model_construct"]["completed"] for b in branches)
        contexts = sum(b["runtime"]["counters"]["context"]["completed"] for b in branches)
        require(controls <= 687 and native <= 17175 and native == controls * 25, "Original phase budgets")
        require(models <= 18 and contexts <= 18, "Original model and context budgets")
        if phase == "checker":
            require(
                all(
                    b["candidate_policy_imported"] is False and b["runtime"]["newly_selected_commands"] == 0
                    for b in branches
                ),
                "Independent checker consumed saved actions only",
            )
        result.update(
            complete=True,
            passed=True,
            branch_refs=branch_refs,
            sources_unchanged=True,
            control_steps=controls,
            native_steps=native,
            model_initializations=models,
            software_contexts=contexts,
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
        require(not DRY.exists(), "New dry output only")
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
        phase, case, mode = args.worker
        require(phase in ("producer", "checker"), "Registered phase")
        index = int(case)
        branch_key(index, mode)
        worker(phase, index, mode)


if __name__ == "__main__":
    main()
