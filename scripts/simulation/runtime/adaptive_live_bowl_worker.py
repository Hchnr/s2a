"""Run one isolated simulation phase for the bounded adaptive bowl trial."""

import os
import socket
import sys
from functools import partial

import numpy as np

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import adaptive_live_bowl_admission as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_bowl_fixed_h40_admission as first_admission
from scripts.simulation.runtime import reset_bowl_next_reference as reference
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation


def prefix_options(inputs, phase):
    """Compose the unchanged saved-reference prefix without patching its modules."""
    prefix.require(phase in ("producer", "checker"), "A registered simulation phase is required")
    source = inputs["prior"]["base"]["source"]
    base = source["previous"]
    common = dict(
        expected_capture=base["base"]["expected_capture"],
        load_reference=lambda: first_admission.load_reference(base["base"]),
        expected_second_capture=base["capture"],
        load_second_reference=lambda: (base["reference"], base["envelope"]),
    )
    if phase == "producer":
        runner = partial(
            reference.run_producer,
            previous_88_producer=source["previous_88_producer"],
            expected_step88_capture=source["capture88"],
            load_step88_reference=lambda: (source["reference88"], source["envelope88"]),
        )
        common.update(
            baseline=base["base"]["baseline"],
            previous_producer=base["previous_producer"],
            previous_56_producer=source["previous_producer"],
        )
    else:
        runner = partial(
            reference.run_checker,
            expected_step88_capture=source["capture88"],
            load_step88_reference=lambda: (source["reference88"], source["envelope88"]),
        )
    return runner, common


def load_recorded_responses(producer, model_result):
    """Bind every saved channel envelope to the actual model worker output."""
    from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

    runtime = producer["runtime"]
    exchanges, captures = runtime["adaptive_model_exchanges"], runtime["adaptive_captures"]
    rows = model_result["responses"]
    prefix.require(
        len(exchanges) == len(captures) == len(rows) == model_result["responses_sent"]
        and model_result["model_requests_attempted"] == model_result["model_requests_completed"] == len(rows)
        and model_result["denoise_calls_attempted"] == model_result["denoise_calls_completed"] == 35 * len(rows),
        "Every accepted reference must bind one completed actual model request",
    )
    responses = []
    for index, (exchange, capture, row) in enumerate(zip(exchanges, captures, rows, strict=True)):
        request, response = prefix.read(exchange["request_ref"]), prefix.read(exchange["response_ref"])
        actual = repro.response_arrays(prefix.read(row["actual_response_ref"]))
        prefix.require(
            exchange["request_id"] == request["request_id"] == index
            and exchange["step"] == request["step"] == row["step"] == capture["frame_index"]
            and request["image"]["sha256"] == response["image_sha256"] == capture["canvas_png"]["sha256"]
            and row["reference_sent"] is True
            and all(
                exchange.get(key) is True
                for key in ("request_sent", "response_received", "response_accepted", "policy_ready")
            ),
            "The actual model exchange must match its current image and response",
        )
        np.testing.assert_array_equal(actual["action"], response["action"])
        np.testing.assert_array_equal(actual["normalized_action"], response["normalized_action"])
        responses.append(response)
    return responses


def simulation_worker(phase, *, source_refs):
    """Recreate the registered reset and publish complete or partial evidence."""
    prefix.require(phase in ("producer", "checker"), "A registered simulation phase is required")
    output = admission.OUTPUT
    directory = output / phase / "simulation"
    result = dict(
        complete=False, passed=False, control_qualified=False, phase=phase, automatic_retry=False, fresh_cases=0
    )
    channel = None
    try:
        prefix.environment()
        inputs = admission.load_inputs()
        registration = prefix.read(prefix.reference(output / phase / "registration.json"))
        prefix.require(
            registration["phase"] == phase
            and registration["inputs"] == admission.input_receipt(inputs)
            and registration["source_refs"] == source_refs(),
            "The simulation worker requires the exact registered inputs and sources",
        )
        producer = responses = None
        if phase == "checker":
            isolation.checker_isolation()
            producer = prefix.read(registration["producer_branch_ref"])
            model_result = prefix.read(registration["producer_model_ref"])
            prefix.require(
                producer["complete"]
                is producer["passed"]
                is model_result["complete"]
                is model_result["passed"]
                is True,
                "The completed producer and model workers are required",
            )
            responses = load_recorded_responses(producer, model_result)
        else:
            channel = FramedJSONChannel(
                socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])),
                max_bytes=2 * 1024 * 1024,
                timeout_seconds=1750,
            )
        source = inputs["prior"]["base"]["source"]
        base = source["previous"]
        old_runtime = base["base"]["base"]["previous"]["base"]
        cv2 = old_runtime["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        from s2a.research.native_runtime.development_prefix_runtime_bounded import PrefixReplayFailure
        from s2a.research.native_runtime.reset_bounded_runtime import ResetBoundedRuntime
        from scripts.simulation.runtime import adaptive_live_bowl as continuation

        case = base["base"]["case"]
        runtime = ResetBoundedRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )
        runtime_output = directory / "runtime"
        stream_registration = admission.model_registration(inputs, directory)
        runner, options = prefix_options(inputs, phase)

        def run_continuation(env, terminal, observers, receipt, *, progress=None):
            common = dict(
                prefix_runner=runner,
                expected_replan_capture=inputs["expected_capture"],
                registration=stream_registration,
                runtime_output=runtime_output,
                case=case,
                progress=progress,
                **options,
            )
            if phase == "producer":
                return continuation.run_producer(
                    env,
                    terminal,
                    observers,
                    receipt,
                    prior_producer=inputs["prior"]["base"]["producer"],
                    expected_response=inputs["expected_response"],
                    channel=channel,
                    **common,
                )
            return continuation.run_checker(
                env, terminal, observers, receipt, producer=producer, responses=responses, **common
            )

        try:
            actual = runtime.run_feedback(
                case["config"],
                case["record"],
                case["record_xml"],
                runtime_output,
                old_runtime["asset_refs"],
                registered_seed=inputs["config"]["seed"],
                continuation=run_continuation,
                max_control_steps=inputs["config"]["max_control_steps_per_phase"],
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        forbidden = tuple(isolation.FORBIDDEN) + (
            "s2a.research.joint_control.preview.causal_reference_replan",
            "s2a.research.wam.tracking.runtime.adaptive_stream_reference",
        )
        unloaded = all(name not in sys.modules for name in forbidden)
        prefix.require(phase != "checker" or unloaded, "The checker cannot load candidate policies")
        result.update(
            complete=True, passed=True, case=case["case"], runtime=actual, candidate_modules_unloaded=unloaded
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if channel is not None:
            channel.close()
        print(publish(directory / "result.json", result), flush=True)
