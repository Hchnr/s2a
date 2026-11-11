"""Execute the registered six-window current-image WAM request loop."""

import copy
import json
import random
import time

import numpy as np

from s2a.research.wam.tracking.runtime.causal_stream_control import check_stop
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalModelAdmission, model_inputs
from scripts.simulation.runtime import causal_stream_recorded_admission as recorded
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine
from scripts.wam.tracking.preview.live_stream_requests import _check_current, _check_response_identity


def execute_live_requests(
    inputs,
    *,
    channel,
    image_root,
    directory,
    service,
    torch,
    counter,
    overrides,
    domain_reader,
    mask_evidence,
    result,
    max_requests,
):
    """Receive current images, generate exactly the registered references, and reply.

    Model and denoise attempts are recorded before the relevant operation. The
    caller owns process, GPU, preflight and exception publication. No simulator
    or future observation is read by this loop.
    """
    config = inputs["live_config"]
    registration = physical.model_config(inputs, image_root)
    prefix.require(
        type(max_requests) is int
        and max_requests == config.get("max_model_requests") == config.get("max_windows") == 6
        and len(inputs["records"]) == 2,
        "An explicit six-window registration with two historical gates is required",
    )
    registration["max_requests"] = max_requests
    model_admission = CausalModelAdmission(**registration)
    result.setdefault("responses", [])
    result.setdefault("local_requests_received", 0)
    result.setdefault("model_requests_attempted", 0)
    result.setdefault("model_requests_completed", 0)
    result.setdefault("response_send_attempts", 0)
    result.setdefault("responses_sent", 0)
    result["model_sequence_complete"] = False
    result["model_sequence_stop"] = None
    try:
        for index in range(max_requests):
            request = channel.receive()
            if type(request) is dict and request.get("kind") == "stop":
                result["model_sequence_stop"] = check_stop(model_admission, request)
                break
            result["local_requests_received"] += 1
            historical = inputs["records"][index] if index < len(inputs["records"]) else None
            if historical is not None:
                _check_current(request, historical)
            visible = model_admission.admit(request)
            prefix.require(
                set(visible) == {"image_bytes", "task"} and visible["task"] == config["task"],
                "Only current image bytes and the public task are model inputs",
            )
            record = historical or {"step": request["step"], "image_sha256": request["image"]["sha256"]}
            item = directory / f"step{record['step']}"
            item.mkdir()
            image_path = item / "current.png"
            with image_path.open("xb") as stream:
                stream.write(visible["image_bytes"])
            prefix.require(repro.sha256(image_path) == record["image_sha256"], "Copied current image changed")
            image = {
                "step": record["step"],
                "path": str(image_path.relative_to(prefix.REPO)),
                "sha256": record["image_sha256"],
            }
            model_request = engine.make_request(inputs["template"], image)
            repro.write_new(item / "request.json", model_request)
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            started = time.monotonic()
            sample, batch = service.prepare(overrides.model_validate(copy.deepcopy(model_request)), item)
            domain = domain_reader(service, sample, batch)
            caption = json.loads(batch["ai_caption"][0])
            prefix.require(
                domain.is_libero
                and caption["task_title"] == caption["command"] == config["task"]
                and caption["task_plan"] == [],
                "Actual WAM conditioning differs from the current public task",
            )
            repro.write_new(item / "conditioning.json", {"caption": caption, "image_sha256": image["sha256"]})
            result["model_requests_attempted"] += 1
            with torch.inference_mode(), counter.scope(model_request["name"], 35):
                response_object = service.execute(sample, batch)
                result["model_requests_completed"] += 1
            response = response_object.model_dump(mode="json")
            repro.write_new(item / "response.json", response)
            row = {
                "step": record["step"],
                "actual_response_ref": prefix.reference(item / "response.json"),
                "model_request_ref": prefix.reference(item / "request.json"),
                "historical_response_sha256": historical["response_sha256"] if historical else None,
                "model_elapsed_seconds": time.monotonic() - started,
                "comparison": None,
                "reference_sent": False,
            }
            result["responses"].append(row)
            arrays = repro.response_arrays(response)
            _check_response_identity(
                response,
                model_request=model_request,
                task=config["task"],
                checkpoint=inputs["prepared"]["model_config"]["checkpoint"],
            )
            prefix.require(
                repro.sha256(image_path) == record["image_sha256"]
                and model_inputs(request, image_root=str(image_root.relative_to(prefix.REPO))) == visible,
                "The current image changed during model execution",
            )
            if historical is not None:
                comparisons = {}
                for key, array in arrays.items():
                    expected = np.asarray(historical[key], dtype=np.float64)
                    difference = array - expected
                    comparisons[key] = {
                        "elementwise_equal": bool(np.array_equal(array, expected)),
                        "max_absolute_difference": float(np.max(np.abs(difference))),
                        "different_elements": int(np.count_nonzero(difference)),
                    }
                row["comparison"] = comparisons
                prefix.require(
                    all(value["elementwise_equal"] for value in comparisons.values()),
                    "Live WAM output differs from the audited historical reference",
                )
            expected_calls = (index + 1) * config["denoise_calls_per_request"]
            prefix.require(
                len(counter.calls) == expected_calls and all(call["completed"] for call in counter.calls),
                "Actual denoise calls differ from the registered budget",
            )
            engine.validate_mask_evidence(mask_evidence, index + 1)
            model_admission.complete(request_id=index, denoise_calls=config["denoise_calls_per_request"])
            identity = {
                key: request[key]
                for key in (
                    "version",
                    "session",
                    "request_id",
                    "step",
                    "sim_time_s",
                    "state_sha256",
                    "seed",
                    "model_sha256",
                )
            }
            envelope = {
                **identity,
                "image_sha256": record["image_sha256"],
                "denoise_calls": config["denoise_calls_per_request"],
                "action_dim_mask": [True] * 10 + [False] * 118,
                "action": arrays["action"].tolist(),
                "normalized_action": arrays["normalized_action"].tolist(),
            }
            if historical is not None:
                recorded.gate_response(inputs["records"], request, envelope)
            result["response_send_attempts"] += 1
            channel.send(envelope)
            result["responses_sent"] += 1
            row["reference_sent"] = True
        if result["model_sequence_stop"] is None:
            prefix.require(
                model_admission.phase == "done" and result["responses_sent"] == max_requests,
                "The registered live request sequence must complete exactly once",
            )
            result["model_sequence_complete"] = True
        else:
            prefix.require(
                model_admission.phase == "ready" and result["responses_sent"] == model_admission.admitted_requests,
                "A controlled stop must follow all completed model responses",
            )
    finally:
        result["denoise_calls_attempted"] = len(counter.calls)
        result["denoise_calls_completed"] = sum(bool(call["completed"]) for call in counter.calls)
        result["denoise_evidence"] = counter.calls
        result["mask_evidence"] = mask_evidence
