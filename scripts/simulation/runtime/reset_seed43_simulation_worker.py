"""Run the registered reset feedback or its independent CPU replay."""

import os
import socket
import sys

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation


def worker_inputs(trial, phase):
    """Recheck preflight and the exact bundle or finished producer before physics."""
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One registered physical phase is required")
    registration = prefix.read(prefix.reference(trial.output / phase / "registration.json"))
    inputs = trial.load_inputs(verify_assets=False)
    trial.check_preflight(inputs)
    prefix.require(
        registration["phase"] == phase
        and registration["inputs"] == trial.input_receipt(inputs)
        and registration["source_refs"] == trial.source_refs(),
        "The worker requires the exact preflight inputs and source identities",
    )
    if phase == "producer":
        bundle_ref = registration["model_bundle_ref"]
        prefix.require(
            bundle_ref == prefix.reference(trial.output / "producer/model/registered_inputs.json")
            and prefix.read(bundle_ref) == trial.model_bundle(inputs)
            and registration["producer_branch_ref"] is registration["producer_model_ref"] is None,
            "The producer requires its exact registered model bundle",
        )
    else:
        previous = prefix.read(prefix.reference(trial.output / "producer/result.json"))
        prefix.require(
            previous["complete"]
            is previous["passed"]
            is previous["sources_unchanged"]
            is previous["gpu_released"]
            is True
            and registration["producer_branch_ref"] == previous["branch_ref"]
            and registration["producer_model_ref"] == previous["model_ref"]
            and registration["model_bundle_ref"] is None,
            "The checker requires the closed producer and its exact model and branch receipts",
        )
    return inputs, registration


def run_worker(trial, phase):
    """Execute one source-bound producer or independently replay its actions."""
    inputs, registration = worker_inputs(trial, phase)
    directory = trial.output / phase / "simulation"
    result = dict(
        complete=False, passed=False, control_qualified=False, phase=phase, automatic_retry=False, fresh_cases=0
    )
    channel = None
    try:
        case = inputs["case"]
        if phase == "checker":
            isolation.checker_isolation()
            producer = prefix.read(registration["producer_branch_ref"])
            model = prefix.read(registration["producer_model_ref"])
            prefix.require(
                producer["complete"] is producer["passed"] is model["complete"] is model["passed"] is True
                and model["gpu_uuid"] == trial.gpu_uuid
                and model["model_initialization_attempts"] == model["model_initializations"] == 1
                and type(model["responses_sent"]) is int
                and model["responses_sent"] in (0, 1)
                and model["model_requests_attempted"] == model["model_requests_completed"] == model["responses_sent"]
                and model["denoise_calls_attempted"]
                == model["denoise_calls_completed"]
                == 35 * model["responses_sent"],
                "The independent checker requires closed producer and model receipts",
            )
            model_response = (
                prefix.read(model["responses"][0]["actual_response_ref"]) if model["responses_sent"] else None
            )
        else:
            producer = model_response = None
            channel = FramedJSONChannel(
                socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])), max_bytes=2 * 1024 * 1024, timeout_seconds=1750
            )
        cv2 = inputs["base"]["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.reset_two_window_runtime import ResetTwoWindowRuntime

        adapter = ResetTwoWindowRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )
        runtime_output = directory / "runtime"
        model_registration = trial.model_registration(inputs, trial.output / "producer/simulation/runtime")

        def continuation(env, terminal, observers, receipt, *, progress=None):
            if phase == "producer":
                from scripts.simulation.runtime.reset_live_two_window import run_producer

                return run_producer(
                    env,
                    terminal,
                    observers,
                    receipt,
                    case=case,
                    baseline=inputs["baseline_periods"],
                    registration=model_registration,
                    channel=channel,
                    runtime_output=runtime_output,
                    progress=progress,
                )
            from scripts.simulation.runtime.reset_live_two_window_replay_v2 import run_checker

            return run_checker(
                env,
                terminal,
                observers,
                receipt,
                case=case,
                producer=producer,
                registration=model_registration,
                model_response=model_response,
                runtime_output=runtime_output,
                progress=progress,
            )

        try:
            runtime = adapter.run_feedback(
                case["config"],
                case["record"],
                case["record_xml"],
                runtime_output,
                inputs["base"]["asset_refs"],
                registered_seed=case["case"]["sampling_seed"],
                continuation=continuation,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        unloaded = all(name not in sys.modules for name in isolation.FORBIDDEN)
        prefix.require(phase != "checker" or unloaded, "The checker may not load candidate strategies")
        result.update(
            complete=True, passed=True, case=case["case"], runtime=runtime, candidate_modules_unloaded=unloaded
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        cleanup_error = None
        try:
            if channel is not None:
                channel.close()
        except BaseException as error:
            cleanup_error = error
            result.update(
                complete=False, passed=False, cleanup_error={"type": type(error).__name__, "message": str(error)}
            )
        print(publish(directory / "result.json", result), flush=True)
        if cleanup_error is not None:
            raise cleanup_error
