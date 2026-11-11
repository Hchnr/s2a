"""Replay one checked feedback prefix and capture current model-ready RGB."""

import argparse
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import run_continuation
from scripts.simulation.runtime import prepare_ordered_correct_stream as binding
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.simulation.runtime import run_ordered_correct_stream as prior

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/references/ordered_feedback_wam_observation_v2_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/references/ordered_feedback_wam_capture_repair_20261111.md"
ROOT = REPO / "outputs/ordered_feedback_wam_observation_v2_20261111"
DRY = REPO / "outputs/ordered_feedback_wam_observation_v2_dry_20261111/result.json"
QA = REPO / "outputs/qa/ordered_feedback_wam_capture_v2_20261111/result.json"
BRANCHES = ((8, "capture"),)


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            REPO / "plans/v0.0.1/tracking/references/ordered_feedback_wam_observation_20261111.md",
            REPO / "configs/tracking/references/ordered_feedback_wam_observation_20261111.json",
            Path(__file__),
            Path(prior.__file__),
            REPO / "src/s2a/research/native_runtime/guarded_correct_prefix.py",
            REPO / "src/s2a/research/wam/training/native_observations.py",
            REPO / "src/s2a/backends/libero/cameras.py",
            REPO / "tests/research/native_runtime/test_capture_ordered_feedback_observations.py",
        )
    ]


def read_config():
    prefix.environment()
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    fixed = {
        "schema": "s2a.ordered-feedback-wam-observation.v2",
        "output": "outputs/ordered_feedback_wam_observation_v2_20261111",
        "case_index": 8,
        "seed": 42,
        "capture_steps": [80, 88],
        "prefix_steps": 80,
        "replayed_continuation_steps": 8,
        "max_cpu_control_steps": 88,
        "max_cpu_native_steps": 2200,
        "previous_cpu_control_steps": 80,
        "previous_cpu_native_steps": 2000,
        "gpu_uuid": "GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf",
        "model_initializations": 1,
        "requests": 2,
        "denoise_calls": 70,
        "world_model_seed": 42,
        "worker_timeout_seconds": 1800,
        "fresh_cases": 0,
        "fits": 0,
        "automatic_retry": False,
    }
    prefix.require(
        all(config.get(k) == v and type(config.get(k)) is type(v) for k, v in fixed.items()),
        "Fixed observation capture and two-request WAM budget required",
    )
    return ref, config


def configured_ref(config, name):
    ref = {"path": str(REPO / config[name]), "sha256": config[name + "_sha256"]}
    prefix.checked(ref)
    return ref


def load_inputs():
    config_ref, config = read_config()
    inputs = binding.load_inputs()
    refs = {
        key: configured_ref(config, key)
        for key in (
            "producer",
            "producer_summary",
            "checker_summary",
            "model_preflight",
            "historical_request",
            "previous_failed",
            "previous_failed_branch",
        )
    }
    producer, summary, checker, model = (
        prefix.read(refs[key]) for key in ("producer", "producer_summary", "checker_summary", "model_preflight")
    )
    prefix.require(
        producer["complete"] is producer["passed"] is summary["complete"] is summary["passed"] is True
        and checker["complete"] is checker["passed"] is model["complete"] is True
        and summary["branch_refs"] == [refs["producer"]]
        and summary["counts"]["continuation_steps"] == checker["counts"]["continuation_steps"] == 16
        and producer["runtime"]["continuation"]["steps_completed"] == 16
        and producer["runtime"]["continuation"]["stop"] is None
        and model["gpu_uuid"] == config["gpu_uuid"],
        "A complete independently replayed correct-feedback source and model preflight are required",
    )
    failure = prefix.read(refs["previous_failed"])
    failed_branch = prefix.read(refs["previous_failed_branch"])
    worker_ref = failure["workers_ref"]
    prefix.checked(worker_ref)
    workers = prefix.read(worker_ref)["workers"]
    failed_runtime = failed_branch["runtime"]
    prefix.require(
        failure["complete"] is failure["passed"] is False
        and failure["available_branch_refs"] == [refs["previous_failed_branch"]]
        and failure["world_model_calls"] == 0
        and failed_branch["error"]["type"] == "PrefixReplayFailure"
        and "path outside approved roots" in failed_branch["error"]["message"]
        and failed_runtime["control_steps_completed"] == config["previous_cpu_control_steps"]
        and failed_runtime["counters"]["mj_step"]["completed"] == config["previous_cpu_native_steps"]
        and failed_runtime["captures"] == []
        and failed_runtime["world_model_calls"] == 0
        and failed_runtime["cleanup_complete"] is True
        and len(workers) == 1
        and workers[0]["returncode"] == 1
        and workers[0]["actual_wait"] is workers[0]["reaped"] is True
        and workers[0]["signal_sent"] is False,
        "The recorded failed capture and its consumed budget must remain verifiable",
    )
    periods = producer["runtime"]["continuation"]["periods"][:8]
    prefix.require(
        len(periods) == 8 and [row["step"] for row in periods] == list(range(80, 88)),
        "The original eight actually executed continuation commands are required",
    )
    consumed = dict(inputs["source_sha256"])
    for ref in [config_ref, *refs.values(), worker_ref]:
        prefix.checked(ref, consumed)
    return {
        "config_ref": config_ref,
        "config": config,
        "refs": refs,
        "prior": inputs,
        "periods": periods,
        "producer": producer,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "refs": inputs["refs"],
        "prior": binding.input_receipt(inputs["prior"]),
        "source_sha256": inputs["source_sha256"],
    }


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered capture branch required")
    return f"case_{index:02d}_{mode}"


def render_registered_endpoint(observers, env, step, sample, endpoint, runtime_output):
    """Place the renderer's entire output inside its guarded runtime root."""
    from s2a.research.native_runtime.bootstrap import _safe_path

    prefix.require(step in (80, 88), "Only registered capture steps are allowed")
    runtime_output = Path(runtime_output)
    _safe_path(runtime_output / f"anchor{step:04d}", (runtime_output,), exists=False)
    return observers.render_anchor(env, step, sample, endpoint, runtime_output)


def worker(phase, index, mode):
    prefix.environment()
    prefix.require(phase == "producer", "Only the registered read-only replay phase is allowed")
    key = branch_key(index, mode)
    directory = ROOT / phase / key
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        registration["inputs"] == input_receipt(inputs) and registration["source_refs"] == source_refs(),
        "Frozen capture input and source identities required",
    )
    prior.checker_isolation()
    previous = inputs["prior"]["previous"]
    expected = inputs["periods"]
    result = {"complete": False, "passed": False, "control_qualified": False, "automatic_retry": False}
    try:
        cv2 = previous["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.guarded_correct_prefix import GuardedCorrectTask2Seed42PrefixRuntime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            prior.checker_isolation()
            recorded = inputs["producer"]["runtime"]["prefix_inner_periods"]
            prefix.require(len(receipt["prefix_inner_periods"]) == len(recorded) == 80, "Every prefix sample required")
            for actual, earlier in zip(receipt["prefix_inner_periods"], recorded, strict=True):
                common.check_prefix_inner(actual, earlier)
            captures = receipt["captures"] = []
            captures.append(
                render_registered_endpoint(
                    observers, env, 80, terminal["sample"], terminal["endpoint_receipt"], directory / "runtime"
                )
            )

            def choose(_q, _limits, _jacobian, _previous, offset):
                return {"command": np.asarray(expected[offset]["command"]), "newly_selected": False}

            def completed(actual_env, _command, offset):
                score = prior.score_opening(actual_env, inputs["prior"], offset)
                prior.compare_score(score, expected[offset]["completed_feedback"]["score"])
                if offset == 7:
                    sample, endpoint = observers.endpoint(actual_env, 88)
                    captures.append(
                        render_registered_endpoint(observers, actual_env, 88, sample, endpoint, directory / "runtime")
                    )
                return {"score": score}

            return run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=80,
                previous_command=previous["commands"][-1],
                targets=inputs["prior"]["targets"][0][:8],
                tool=previous["tool"],
                target_object=inputs["prior"]["config"]["target_object"],
                choose_action=choose,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=completed,
                progress=progress,
            )

        try:
            runtime = GuardedCorrectTask2Seed42PrefixRuntime().run(
                previous["runtime_config"],
                previous["record"],
                previous["record_xml"],
                previous["commands"],
                previous["saved"],
                previous["receipts"],
                directory / "runtime",
                previous["asset_refs"],
                previous["controller"],
                continuation=continuation,
                continuation_budget=8,
                registered_seed=42,
                capture_final_prefix_inner=True,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        prior.checker_isolation()
        prefix.require(
            runtime["continuation"]["steps_completed"] == 8
            and [item["frame_index"] for item in runtime["captures"]] == [80, 88]
            and runtime["newly_selected_commands"] == 0,
            "Two images at independently replayed actual endpoints required",
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
        "Bounded complete replay and two current image captures required",
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
        worker_module="scripts.simulation.runtime.capture_ordered_feedback_observations",
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
