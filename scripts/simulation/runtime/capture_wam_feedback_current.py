"""Replay a checked WAM feedback window and capture its current endpoint images."""

import argparse
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import run_continuation
from scripts.simulation.runtime import capture_ordered_feedback_observations as capture
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.simulation.runtime import run_saved_wam_feedback as study

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/references/wam_feedback_current_observation_v2_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/references/wam_feedback_current_capture_repair_20261111.md"
ROOT = REPO / "outputs/wam_feedback_current_observation_v2_20261111"
DRY = REPO / "outputs/wam_feedback_current_observation_v2_dry_20261111/result.json"
QA = REPO / "outputs/qa/wam_feedback_current_observation_v2_20261111/result.json"
BRANCHES = ((8, "capture"),)


def source_refs():
    return study.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            REPO / "configs/tracking/references/wam_feedback_current_observation_20261111.json",
            REPO / "plans/v0.0.1/tracking/references/wam_feedback_current_observation_20261111.md",
            Path(__file__),
            REPO / "tests/research/native_runtime/test_capture_wam_feedback_current.py",
        )
    ]


def read_config():
    prefix.environment()
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    fixed = {
        "schema": "s2a.wam-feedback-current-observation.v2",
        "output": str(ROOT.relative_to(REPO)),
        "case_index": 8,
        "seed": 42,
        "capture_steps": [80, 88],
        "prefix_steps": 80,
        "replayed_continuation_steps": 8,
        "max_cpu_control_steps": 88,
        "max_cpu_native_steps": 2200,
        "previous_cpu_control_steps": 88,
        "previous_cpu_native_steps": 2200,
        "worker_timeout_seconds": 1800,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "fits": 0,
        "automatic_retry": False,
    }
    prefix.require(
        all(config.get(key) == value and type(config.get(key)) is type(value) for key, value in fixed.items()),
        "Fixed single replay and current observation budget required",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    base = study.load_inputs()
    refs = {
        key: study.binding.configured_ref(config, key)
        for key in ("producer", "producer_branch", "checker", "metrics", "previous_failure", "previous_failure_branch")
    }
    producer, branch, checker, metrics = (
        prefix.read(refs[key]) for key in ("producer", "producer_branch", "checker", "metrics")
    )
    periods = branch["runtime"]["continuation"]["periods"]
    prefix.require(
        producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
        and checker["complete"] is checker["passed"] is checker["sources_unchanged"] is True
        and producer["branch_refs"] == [refs["producer_branch"]]
        and producer["counts"]["continuation_steps"] == checker["counts"]["continuation_steps"] == 8
        and branch["complete"] is branch["passed"] is metrics["complete"] is True
        and branch["runtime"]["continuation"]["stop"] is None
        and metrics["original_wam_worker_success"] is metrics["initial_noise_ledger_available"] is False
        and len(periods) == 8
        and [row["step"] for row in periods] == list(range(80, 88)),
        "One completed and independently replayed partial-WAM feedback window required",
    )
    failure = prefix.read(refs["previous_failure"])
    failed_branch = prefix.read(refs["previous_failure_branch"])
    worker_ref = failure["workers_ref"]
    workers = prefix.read(worker_ref)["workers"]
    failed_runtime = failed_branch["runtime"]
    prefix.require(
        failure["complete"] is failure["passed"] is False
        and failure["available_branch_refs"] == [refs["previous_failure_branch"]]
        and failure["world_model_calls"] == 0
        and failed_branch["error"]["type"] == "PrefixReplayFailure"
        and "missing 1 required positional argument: 'runtime_output'" in failed_branch["error"]["message"]
        and failed_runtime["control_steps_completed"] == config["previous_cpu_control_steps"]
        and failed_runtime["counters"]["mj_step"]["completed"] == config["previous_cpu_native_steps"]
        and len(failed_runtime["continuation_partial"]["periods"]) == 8
        and [item["frame_index"] for item in failed_runtime["captures"]] == [80]
        and failed_runtime["world_model_calls"] == 0
        and failed_runtime["cleanup_complete"] is True
        and len(workers) == 1
        and workers[0]["returncode"] == 1
        and workers[0]["actual_wait"] is workers[0]["reaped"] is True
        and workers[0]["signal_sent"] is False,
        "The original failed capture and its consumed budget must remain verifiable",
    )
    consumed = dict(base["source_sha256"])
    for ref in (config_ref, *refs.values(), worker_ref):
        prefix.checked(ref, consumed)
    return {
        "config_ref": config_ref,
        "config": config,
        "refs": refs,
        "base": base,
        "branch": branch,
        "periods": periods,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "refs": inputs["refs"],
        "base": study.input_receipt(inputs["base"]),
        "source_sha256": inputs["source_sha256"],
    }


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One actual feedback capture branch required")
    return f"case_{index:02d}_{mode}"


def render_completed_endpoint(observers, env, expected_period, runtime_output):
    sample, endpoint = observers.endpoint(env, 88)
    expected = expected_period["endpoint_receipt"]
    prefix.require(
        endpoint["sim_state_sha256"] == expected["sim_state_sha256"] and endpoint["sim_time"] == expected["sim_time"],
        "Step88 must be the actual recorded WAM feedback endpoint",
    )
    return capture.render_registered_endpoint(observers, env, 88, sample, endpoint, runtime_output)


def worker(phase, index, mode):
    prefix.environment()
    prefix.require(phase == "producer", "Only the registered replay phase is allowed")
    key = branch_key(index, mode)
    directory = ROOT / phase / key
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        registration["inputs"] == input_receipt(inputs) and registration["source_refs"] == source_refs(),
        "Frozen actual feedback capture inputs and sources required",
    )
    study.prior.checker_isolation()
    base, expected = inputs["base"], inputs["periods"]
    old = base["base"]["previous"]
    result = {"complete": False, "passed": False, "control_qualified": False, "automatic_retry": False}
    try:
        cv2 = old["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.guarded_correct_prefix import GuardedCorrectTask2Seed42PrefixRuntime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            study.prior.checker_isolation()
            recorded = inputs["branch"]["runtime"]["prefix_inner_periods"]
            prefix.require(len(receipt["prefix_inner_periods"]) == len(recorded) == 80, "Every prefix sample required")
            for actual, earlier in zip(receipt["prefix_inner_periods"], recorded, strict=True):
                common.check_prefix_inner(actual, earlier)
            images = receipt["captures"] = []
            first = capture.render_registered_endpoint(
                observers, env, 80, terminal["sample"], terminal["endpoint_receipt"], directory / "runtime"
            )
            study.validate_live_capture(base["source_capture"], first)
            images.append(first)
            policy_inputs = study.model_policy_inputs(base, base["anchor"])

            def choose(_q, _limits, _jacobian, _previous, offset):
                return {"command": np.asarray(expected[offset]["command"]), "newly_selected": False}

            def completed(actual_env, _command, offset):
                score = study.prior.score_opening(actual_env, policy_inputs, offset)
                study.prior.compare_score(score, expected[offset]["completed_feedback"]["score"])
                if offset == 7:
                    images.append(
                        render_completed_endpoint(observers, actual_env, expected[offset], directory / "runtime")
                    )
                return {"score": score}

            return run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=80,
                previous_command=old["commands"][-1],
                targets=base["targets"][:8],
                tool=old["tool"],
                target_object=base["base"]["config"]["target_object"],
                choose_action=choose,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=completed,
                progress=progress,
                continuation_limit=8,
            )

        try:
            runtime = GuardedCorrectTask2Seed42PrefixRuntime().run(
                old["runtime_config"],
                old["record"],
                old["record_xml"],
                old["commands"],
                old["saved"],
                old["receipts"],
                directory / "runtime",
                old["asset_refs"],
                old["controller"],
                continuation=continuation,
                continuation_budget=8,
                registered_seed=42,
                capture_final_prefix_inner=True,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        study.prior.checker_isolation()
        prefix.require(
            runtime["continuation"]["steps_completed"] == 8
            and [item["frame_index"] for item in runtime["captures"]] == [80, 88]
            and runtime["newly_selected_commands"] == 0,
            "Two current images from replayed actual feedback required",
        )
        result.update(complete=True, passed=True, runtime=runtime, candidate_modules_unloaded=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def summarize_phase(branches, phase, config):
    prefix.require(phase == "producer" and len(branches) == 1, "One capture phase required")
    branch, runtime = branches[0], branches[0]["runtime"]
    counts = {
        "control_steps": runtime["control_steps_completed"],
        "native_steps": runtime["counters"]["mj_step"]["completed"],
        "prefix_inner_samples": sum(len(period["samples"]) for period in runtime["prefix_inner_periods"]),
        "continuation_steps": runtime["continuation"]["steps_completed"],
        "captured_steps": [item["frame_index"] for item in runtime["captures"]],
        "newly_selected_commands": runtime["newly_selected_commands"],
    }
    prefix.require(
        branch["complete"] is branch["passed"] is branch["candidate_modules_unloaded"] is True
        and counts["control_steps"] == config["max_cpu_control_steps"]
        and counts["native_steps"] == config["max_cpu_native_steps"]
        and counts["prefix_inner_samples"] == 2000
        and counts["continuation_steps"] == 8
        and counts["captured_steps"] == [80, 88]
        and counts["newly_selected_commands"] == 0
        and runtime["cleanup_complete"] is True,
        "Complete bounded replay and current observations required",
    )
    return counts


def dry_run():
    inputs = load_inputs()
    print(
        publish(
            DRY,
            {
                "complete": True,
                "passed": True,
                "inputs": input_receipt(inputs),
                "source_refs": source_refs(),
                "world_model_calls": 0,
                "physics_steps": 0,
                "fresh_cases": 0,
            },
        )
    )


def execute():
    return registered_feedback_phase.execute_phase(
        "producer",
        root=ROOT,
        qa_path=QA,
        dry_path=DRY,
        branches=BRANCHES,
        worker_module="scripts.simulation.runtime.capture_wam_feedback_current",
        branch_key=branch_key,
        load_inputs=load_inputs,
        input_receipt=input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="config",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--capture", action="store_true")
    group.add_argument("--worker", nargs=3, metavar=("PHASE", "CASE", "BRANCH"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.capture:
        execute()
    else:
        phase, index, mode = args.worker
        worker(phase, int(index), mode)


if __name__ == "__main__":
    main()
