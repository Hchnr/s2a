"""Recheck the first recorded WAM refresh across six reset-origin cases."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result
from s2a.references.action_reference import decode_action_targets
from s2a.research.native_runtime.matched_reference_probe import first_refresh_prefix, run_matched_reference_probe
from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import replay_multitask_wrench as admission
from scripts.simulation.runtime import reset_live_two_window_replay as captures
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation
from scripts.tracking.response_models.audit_refresh_margin_events import check_signal
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/multitask_margin_recheck_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/multitask_margin_recheck_20261116.md"
OUTPUT = ROOT / "outputs/multitask_margin_recheck_20261116"
DRY = ROOT / "outputs/multitask_margin_recheck_dry_20261116.json"
QA = ROOT / "outputs/qa/multitask_margin_recheck_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.run_multitask_margin_recheck"
CANDIDATE_MODULE = "s2a.research.native_runtime.deferred_margin_recheck"
EXPECTED_CASES = [
    dict(case_index=0, sampling_seed=42, trigger_step=26, trigger_reason="quality_margin_consumed"),
    dict(case_index=2, sampling_seed=43, trigger_step=17, trigger_reason="quality_margin_consumed"),
    dict(case_index=4, sampling_seed=42, trigger_step=143, trigger_reason="quality_margin_consumed"),
    dict(case_index=6, sampling_seed=43, trigger_step=17, trigger_reason="no_audited_second_pose"),
    dict(case_index=8, sampling_seed=42, trigger_step=17, trigger_reason="quality_margin_consumed"),
    dict(case_index=10, sampling_seed=43, trigger_step=17, trigger_reason="quality_margin_consumed"),
]


def validate_config(config):
    fixed = dict(
        schema="s2a.multitask-margin-recheck.v1",
        output="outputs/multitask_margin_recheck_20261116",
        development_review=dict(
            path="outputs/reset_multitask_adaptive_complete_audit_20261113/result.json",
            sha256="7fe0f7d17ddea1913ffab2f5a4243c5b8e4b6a5def9633269ebec95efc82f8fe",
        ),
        cases=EXPECTED_CASES,
        max_deferred_triggers_per_case=1,
        max_extra_periods_per_case=1,
        max_total_control_steps_per_phase=242,
        max_total_native_steps_per_phase=6050,
        max_workers_per_phase=6,
        max_models_per_phase=18,
        max_contexts_per_phase=18,
        max_quality_responses_per_period=320,
        max_baseline_responses_per_period=480,
        worker_timeout_seconds=3600,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    if (
        not isinstance(config, dict)
        or set(config) != set(fixed)
        or any(config[key] != value or type(config[key]) is not type(value) for key, value in fixed.items())
        or any(
            type(row[key]) is not int
            for row in config["cases"]
            for key in ("case_index", "sampling_seed", "trigger_step")
        )
    ):
        raise ValueError("The fixed six-case first-trigger roster and physical/model budgets are required")


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            Path(admission.__file__).resolve(),
            ROOT / "src/s2a/research/native_runtime/matched_reference_probe.py",
            ROOT / "src/s2a/research/native_runtime/deferred_margin_recheck.py",
            ROOT / "src/s2a/research/native_runtime/development_continuation.py",
            ROOT / "scripts/tracking/response_models/audit_refresh_margin_events.py",
            ROOT / "tests/research/native_runtime/test_matched_reference_probe.py",
            ROOT / "tests/research/native_runtime/test_deferred_margin_recheck.py",
            ROOT / "tests/research/native_runtime/test_multitask_margin_recheck.py",
        )
    ]


def prepare(spec):
    """Admit history and references only through the registered first refresh."""
    inputs, producer, model, _bundle, row, registration, changes = admission.admit(spec["case_index"])
    sequence, runtime = producer["runtime"]["continuation"], producer["runtime"]
    selected = first_refresh_prefix(
        sequence["adaptive"]["segments"], expected_step=spec["trigger_step"], expected_reason=spec["trigger_reason"]
    )
    initial = copy.deepcopy(sequence["first_window"]["periods"])
    prefix.require(
        len(initial) == 8
        and row["case_index"] == spec["case_index"]
        and inputs["case"]["case"]["sampling_seed"] == spec["sampling_seed"]
        and sum(len(window["periods"]) for window in selected["windows"]) + 8 == spec["trigger_step"],
        "The original case, seed and complete reset-origin prefix must match",
    )
    comparison.compare_periods(initial, inputs["baseline_periods"])
    final = selected["windows"][-1]["periods"][-1]
    checked = check_signal(final, selected["trigger"])
    prefix.require(
        checked["reason"] == spec["trigger_reason"], "The recorded first trigger must be independently checked"
    )
    count = len(selected["windows"])
    saved_captures = copy.deepcopy(runtime["adaptive_captures"][:count])
    exchanges = copy.deepcopy(runtime["adaptive_model_exchanges"][:count])
    responses = copy.deepcopy(model["responses"][:count])
    prefix.require(
        len(saved_captures) == len(exchanges) == len(responses) == count
        and all(
            window["start_step"] == capture["frame_index"] == exchange["step"] == response["step"]
            for window, capture, exchange, response in zip(
                selected["windows"], saved_captures, exchanges, responses, strict=True
            )
        ),
        "Only already-generated references from the completed prefix may be admitted",
    )
    refs = [admission.SOURCE, row["producer_ref"], row["model_ref"], row["checker_ref"], row["original_owner_ref"]]
    for exchange, response in zip(exchanges, responses, strict=True):
        refs.extend((exchange["request_ref"], exchange["response_ref"], response["actual_response_ref"]))
    for ref in refs:
        prefix.checked(ref)
    identity = dict(
        spec=copy.deepcopy(spec),
        inputs=admission.admission.input_receipt(inputs),
        source_refs=refs,
        original_registration_refs=registration["source_refs"],
        documented_source_replacements=changes,
        reference_steps=[window["start_step"] for window in selected["windows"]],
    )
    return dict(
        inputs=inputs,
        initial=initial,
        windows=selected["windows"],
        trigger=selected["trigger"],
        captures=saved_captures,
        exchanges=exchanges,
        responses=responses,
        identity=identity,
    )


def saved_reference(capture, exchange, response, tool, *, step, seed):
    """Decode a saved H40 only after its original current image/state is matched."""
    request, envelope, raw = (
        prefix.read(exchange["request_ref"]),
        prefix.read(exchange["response_ref"]),
        prefix.read(response["actual_response_ref"]),
    )
    endpoint = capture["current_endpoint"]
    prefix.require(
        capture["frame_index"] == exchange["step"] == request["step"] == envelope["step"] == response["step"] == step
        and request["sim_time_s"] == envelope["sim_time_s"] == endpoint["sim_time"]
        and request["state_sha256"] == envelope["state_sha256"] == endpoint["sim_state_sha256"]
        and request["image"]["sha256"] == envelope["image_sha256"] == capture["canvas_png"]["sha256"]
        and request["seed"] == envelope["seed"] == seed
        and envelope["denoise_calls"] == 35
        and response["reference_sent"] is True
        and all(exchange.get(key) is True for key in ("request_sent", "response_received", "response_accepted")),
        "The original model output must bind the current image, state, clock and seed",
    )
    arrays = repro.response_arrays(raw)
    np.testing.assert_array_equal(arrays["action"], envelope["action"])
    np.testing.assert_array_equal(arrays["normalized_action"], envelope["normalized_action"])
    anchor = np.asarray(capture["current_source_pose"], dtype=float) @ np.asarray(tool, dtype=float)
    return decode_action_targets(anchor, arrays["action"], horizon=40)


def checker_isolation():
    isolation.checker_isolation()
    prefix.require(CANDIDATE_MODULE not in sys.modules, "The checker cannot load the diagnostic action selector")


def dry_run():
    prefix.environment()
    original.require_no_simulator_import()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    cases = [prepare(spec) for spec in config["cases"]]
    checker_isolation()
    print(
        publish_result(
            DRY,
            dict(
                complete=True,
                passed=True,
                config_ref=config_ref,
                identities=[case["identity"] for case in cases],
                source_refs=source_refs(),
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def worker(phase, index):
    prefix.require(phase in ("producer", "checker") and index in admission.INDICES, "One fixed case is required")
    print(json.dumps(dict(event="worker_started", pid=os.getpid(), phase=phase, case_index=index)), flush=True)
    directory = OUTPUT / phase / f"case{index:02d}" / "worker"
    directory.mkdir(parents=True, exist_ok=False)
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))

    def execute():
        spec = next(spec for spec in EXPECTED_CASES if spec["case_index"] == index)
        data = prepare(spec)
        prefix.require(
            registration["phase"] == phase
            and index in registration["eligible_cases"]
            and registration["identities"][str(index)] == data["identity"]
            and registration["source_refs"] == source_refs(),
            "The registered case, historical inputs and source identity must match",
        )
        recorded = None
        if phase == "checker":
            checker_isolation()
            produced = prefix.read(registration["producer_branch_refs"][str(index)])
            validate_branch(produced, "producer", spec)
            recorded = produced["runtime"]["continuation"]["probe"]
        inputs, case = data["inputs"], data["inputs"]["case"]
        cv2 = inputs["base"]["cv2_admission"]
        cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        from s2a.research.native_runtime.reset_bounded_runtime import ResetBoundedRuntime

        adapter = ResetBoundedRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )
        observed = []

        def continuation(env, terminal, observers, receipt, *, progress=None):
            def load_window(actual_env, current, position, step):
                sample, endpoint = current["sample"], current["endpoint_receipt"]
                identity = dict(
                    endpoint_step=step,
                    sim_time=endpoint["sim_time"],
                    sim_state_sha256=endpoint["sim_state_sha256"],
                )
                scope = {}
                actual = render_with_two_torch_threads(
                    observers.render_anchor, scope, actual_env, step, sample, identity, directory / "runtime"
                )
                captures.compare_capture(data["captures"][position], actual)
                observed.append(actual)
                return saved_reference(
                    actual,
                    data["exchanges"][position],
                    data["responses"][position],
                    case["tool"],
                    step=step,
                    seed=spec["sampling_seed"],
                )

            return run_matched_reference_probe(
                env,
                terminal,
                observers,
                receipt,
                phase=phase,
                case=case,
                initial_periods=data["initial"],
                windows=data["windows"],
                expected_trigger=data["trigger"],
                load_window=load_window,
                score_gripper=original.measured_gripper,
                compare_score=original.compare_score,
                recorded_probe=recorded,
                progress=progress,
            )

        def progress(receipt):
            counters = receipt.get("counters", {})
            print(
                json.dumps(
                    dict(
                        event="physical_progress",
                        pid=os.getpid(),
                        phase=phase,
                        case_index=index,
                        control_steps_attempted=receipt.get("control_steps_attempted"),
                        control_steps_completed=receipt.get("control_steps_completed"),
                        native_steps=counters.get("mj_step"),
                        models=counters.get("model_construct"),
                        contexts=counters.get("context"),
                    )
                ),
                flush=True,
            )

        max_steps = spec["trigger_step"] + (spec["trigger_reason"] == "quality_margin_consumed")
        actual = adapter.run_feedback(
            case["config"],
            case["record"],
            case["record_xml"],
            directory / "runtime",
            inputs["base"]["asset_refs"],
            registered_seed=spec["sampling_seed"],
            continuation=continuation,
            max_control_steps=max_steps,
            progress=progress,
        )
        if phase == "checker":
            checker_isolation()
        prefix.require(len(observed) == len(data["windows"]), "Every used reference must have a matched current image")
        return dict(
            complete=True,
            passed=True,
            case=case["case"],
            runtime=actual,
            reference_captures=observed,
            cv2_admission=cv2_receipt,
            candidate_modules_unloaded=phase == "checker",
            fresh_cases=0,
        )

    managed.run_managed_physical_worker(phase=phase, result_path=directory / "result.json", execute=execute)


def periods(branch):
    continuation = branch["runtime"]["continuation"]
    return (
        continuation["initial"]["periods"]
        + [period for window in continuation["windows"] for period in window["continuation"]["periods"]]
        + continuation["probe"]["periods"]
    )


def validate_branch(branch, phase, spec):
    actual = branch["runtime"]
    continuation = actual["continuation"]
    rows, probe = periods(branch), continuation["probe"]
    allowed = spec["trigger_reason"] == "quality_margin_consumed"
    prefix.require(
        branch["complete"] is branch["passed"] is actual["complete"] is actual["passed"] is True
        and branch["phase"] == phase
        and branch["case"]["case_index"] == spec["case_index"]
        and branch["case"]["sampling_seed"] == spec["sampling_seed"]
        and actual["reset_observation_verified"] is True
        and actual["prefix_control_steps"] == 0
        and actual["world_model_calls"] == branch["fresh_cases"] == 0
        and actual["control_steps_attempted"] == actual["control_steps_completed"] == len(rows)
        and len(rows) == spec["trigger_step"] + len(probe["periods"])
        and len(probe["periods"]) <= int(allowed)
        and actual["counters"]["mj_step"]["attempted"] == actual["counters"]["mj_step"]["completed"] == 25 * len(rows)
        and actual["counters"]["model_construct"]["attempted"]
        == actual["counters"]["model_construct"]["completed"]
        == actual["counters"]["context"]["attempted"]
        == actual["counters"]["context"]["completed"]
        == 3
        and actual["cleanup_complete"] is True
        and len(branch["reference_captures"]) == len(continuation["windows"]),
        "A closed bounded reset-origin branch with exact actual physical counts is required",
    )
    if phase == "producer":
        events = continuation["deferred_events"]
        prefix.require(
            actual["newly_selected_commands"] == len(rows)
            and continuation["stop_recomputed"] is True
            and len(events) == int(allowed)
            and (allowed or len(probe["periods"]) == 0),
            "The producer must preserve the negative control and at most one deferral",
        )
        if allowed:
            prefix.require(
                events[0]["current_step"] == spec["trigger_step"]
                and events[0]["original_stop"]["stop"]["trigger"]["reason"] == spec["trigger_reason"],
                "Only the registered first margin signal may be deferred",
            )
        else:
            prefix.require(
                probe["stop"]["reason"] == "quality_refresh_required"
                and probe["stop"]["trigger"]["reason"] == "no_audited_second_pose",
                "A missing second-pose audit must stop before dispatch",
            )
        for period in probe["periods"]:
            verified = period["decision"]["quality"]["verification"]
            prefix.require(
                verified["passed"] is verified["first_pose_caps_satisfied"] is True,
                "Any dispatched probe requires the original independent witness",
            )
    else:
        prefix.require(
            branch["candidate_modules_unloaded"] is True
            and actual["newly_selected_commands"] == 0
            and continuation["stop_recomputed"] is False,
            "The checker must replay without loading or selecting the candidate",
        )
    item = probe["periods"][0] if probe["periods"] else None
    return dict(
        control_steps=len(rows),
        native_steps=25 * len(rows),
        prefix_steps=spec["trigger_step"],
        probe_steps=len(probe["periods"]),
        probe_pose_caps_satisfied=None if item is None else item["pose_caps_satisfied"],
        probe_tcp_error=None if item is None else item["tcp_error"],
        probe_minimum_inner_margin_rad=None if item is None else item["minimum_inner_margin_rad"],
        probe_inner_below_buffer_samples=None if item is None else item["inner_below_buffer_samples"],
        probe_task_success=None if item is None else item["task_success"],
        probe_target_grasp=None if item is None else item["target_grasp"],
        producer_stop=probe["stop"] if phase == "producer" else None,
    )


def phase_run(phase):
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One physical phase is required")
    qa_ref, dry_ref = prefix.reference(QA), prefix.reference(DRY)
    qa, dry = prefix.read(qa_ref), prefix.read(dry_ref)
    before = snapshot(ROOT)["sources"]
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and before == prefix.read(qa["source_before_ref"])["sources"]
        and dry["complete"] is dry["passed"] is True
        and dry["physics_steps"] == 0,
        "Passing source-bound CPU QA and zero-physics preflight are required",
    )
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    data = [prepare(spec) for spec in config["cases"]]
    sources = source_refs()
    identities = {str(item["identity"]["spec"]["case_index"]): item["identity"] for item in data}
    prefix.require(
        dry["config_ref"] == config_ref
        and dry["identities"] == [item["identity"] for item in data]
        and dry["source_refs"] == sources,
        "Dry inputs and sources must match",
    )
    producer_refs = {}
    if phase == "checker":
        producer = prefix.read(prefix.reference(OUTPUT / "producer/result.json"))
        prefix.require(
            producer["owner_closed"] is producer["sources_unchanged"] is True, "The producer owner must close"
        )
        eligible = [
            spec["case_index"]
            for spec in config["cases"]
            if producer["cases"][str(spec["case_index"])]["status"] == "closed"
        ]
        producer_refs = {str(index): producer["cases"][str(index)]["branch_ref"] for index in eligible}
    else:
        eligible = [spec["case_index"] for spec in config["cases"]]
    directory = OUTPUT / phase
    directory.mkdir(parents=True, exist_ok=False)
    registration_ref = publish_result(
        directory / "registration.json",
        dict(
            phase=phase,
            identities=identities,
            source_refs=sources,
            qa_ref=qa_ref,
            dry_ref=dry_ref,
            eligible_cases=eligible,
            producer_branch_refs=producer_refs,
            automatic_retry=False,
        ),
    )
    specs = [
        dict(
            key=str(index),
            argv=[sys.executable, "-m", MODULE, "--worker", phase, str(index)],
            log=directory / f"case{index:02d}/worker.log",
        )
        for index in eligible
    ]
    lifecycle = (
        run_owned_workers(specs, cwd=ROOT, timeout_seconds=config["worker_timeout_seconds"])
        if specs
        else dict(workers=[], launch_error=None)
    )
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    lifecycle_ref = publish_result(directory / "workers.json", lifecycle)
    workers = {int(item["key"]): item for item in lifecycle["workers"]}
    outcomes = {}
    result = dict(
        complete=False,
        passed=False,
        owner_closed=False,
        sources_unchanged=False,
        phase=phase,
        registration_ref=registration_ref,
        lifecycle_ref=lifecycle_ref,
        cases=outcomes,
        eligible_cases=eligible,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        for spec in config["cases"]:
            index = spec["case_index"]
            entry = dict(status="not_run", registered_prefix_steps=spec["trigger_step"])
            outcomes[str(index)] = entry
            if index not in eligible:
                entry["reason"] = "producer_not_closed"
                continue
            worker_info = workers.get(index)
            if worker_info is None:
                entry["reason"] = "worker_not_launched"
                continue
            result_path = directory / f"case{index:02d}/worker/result.json"
            if result_path.is_file():
                entry["available_branch_ref"] = prefix.reference(result_path)
            try:
                prefix.require(
                    worker_info["returncode"] == 0
                    and worker_info["actual_wait"] is worker_info["reaped"] is True
                    and worker_info["signal_sent"] is False
                    and result_path.is_file(),
                    "The owned worker must close",
                )
                branch_ref = prefix.reference(result_path)
                branch = prefix.read(branch_ref)
                counts = validate_branch(branch, phase, spec)
                if phase == "checker":
                    produced = prefix.read(producer_refs[str(index)])
                    counts["replay_max_difference"] = comparison.compare_periods(periods(produced), periods(branch))
                entry.update(status="closed", branch_ref=branch_ref, counts=counts)
            except Exception as error:
                entry.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        owner_closed = all(item["actual_wait"] is item["reaped"] is True for item in lifecycle["workers"])
        unchanged = (
            snapshot(ROOT)["sources"] == before
            and source_refs() == sources
            and [prepare(spec)["identity"] for spec in config["cases"]] == [item["identity"] for item in data]
        )
        closed = [item["counts"] for item in outcomes.values() if item["status"] == "closed"]
        known_steps = sum(item["control_steps"] for item in closed)
        known_native = sum(item["native_steps"] for item in closed)
        prefix.require(
            known_steps <= config["max_total_control_steps_per_phase"]
            and known_native <= config["max_total_native_steps_per_phase"],
            "Closed workers must remain within the full registered physical budget",
        )
        result.update(
            owner_closed=owner_closed,
            sources_unchanged=unchanged,
            known_closed_control_steps=known_steps,
            known_closed_native_steps=known_native,
            unaccounted_cases=[int(key) for key, item in outcomes.items() if item["status"] == "failed"],
            complete=owner_closed and unchanged and lifecycle["launch_error"] is None,
            passed=owner_closed
            and unchanged
            and lifecycle["launch_error"] is None
            and len(outcomes) == 6
            and all(item["status"] == "closed" for item in outcomes.values()),
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        print(publish_result(directory / "result.json", result), flush=True)
    if not result["passed"]:
        raise SystemExit("One or more recorded cases did not close successfully")


def audit():
    """Compare every closed producer, checker and original prefix without dropping failures."""
    prefix.environment()
    before = snapshot(ROOT)["sources"]
    config = prefix.read(prefix.reference(CONFIG))
    validate_config(config)
    producer_ref = prefix.reference(OUTPUT / "producer/result.json")
    checker_ref = prefix.reference(OUTPUT / "checker/result.json")
    producer, checker = prefix.read(producer_ref), prefix.read(checker_ref)
    prefix.require(
        producer["owner_closed"]
        is checker["owner_closed"]
        is producer["sources_unchanged"]
        is checker["sources_unchanged"]
        is True,
        "Both physical phase owners must close with unchanged sources",
    )
    registrations = [prefix.read(group["registration_ref"]) for group in (producer, checker)]
    sources = source_refs()
    prefix.require(
        all(registration["source_refs"] == sources for registration in registrations),
        "Both phases must retain the fixed source identities",
    )
    rows = []
    for spec in config["cases"]:
        index = spec["case_index"]
        data = prepare(spec)
        prefix.require(
            all(registration["identities"][str(index)] == data["identity"] for registration in registrations),
            "Both phases must bind the same original prefix and generated references",
        )
        p, c = producer["cases"][str(index)], checker["cases"][str(index)]
        replayed = p["status"] == c["status"] == "closed"
        item = dict(
            spec=spec,
            producer_status=p["status"],
            checker_status=c["status"],
            producer_ref=p.get("branch_ref"),
            checker_ref=c.get("branch_ref"),
            replayed=replayed,
            counts=p.get("counts"),
            replay_max_difference=None,
            original_prefix_max_difference=None,
        )
        if replayed:
            left, right = prefix.read(p["branch_ref"]), prefix.read(c["branch_ref"])
            pc, cc = validate_branch(left, "producer", spec), validate_branch(right, "checker", spec)
            prefix.require(
                pc == p["counts"] and cc["control_steps"] == c["counts"]["control_steps"], "Physical counts differ"
            )
            replay = comparison.compare_periods(periods(left), periods(right))
            prefix.require(replay == c["counts"]["replay_max_difference"], "Independent replay maxima differ")
            original_periods = data["initial"] + [period for window in data["windows"] for period in window["periods"]]
            original_max = comparison.compare_periods(original_periods, periods(left)[: spec["trigger_step"]])
            for expected, actual in zip(left["reference_captures"], right["reference_captures"], strict=True):
                captures.compare_capture(expected, actual)
            item.update(replay_max_difference=replay, original_prefix_max_difference=original_max)
        rows.append(item)
    prefix.require(snapshot(ROOT)["sources"] == before and source_refs() == sources, "Audit sources changed")
    print(
        publish_result(
            OUTPUT / "audit.json",
            dict(
                complete=True,
                passed=all(row["replayed"] for row in rows),
                control_qualified=False,
                registered_cases=6,
                replayed_cases=sum(row["replayed"] for row in rows),
                cases=rows,
                producer_ref=producer_ref,
                checker_ref=checker_ref,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
            ),
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", nargs=2, metavar=("PHASE", "CASE"))
    group.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    prefix.environment()
    if args.dry:
        dry_run()
    elif args.phase:
        phase_run(args.phase)
    elif args.worker:
        worker(args.worker[0], int(args.worker[1]))
    else:
        audit()


if __name__ == "__main__":
    main()
