"""Serve bounded variable-cadence current-image WAM requests."""

import copy
import json
import random
import time
from pathlib import Path

import numpy as np

from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveModelAdmission, check_adaptive_stop
from s2a.research.wam.tracking.runtime.causal_stream_protocol import model_inputs
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine
from scripts.wam.tracking.preview.live_stream_requests import _check_response_identity
from scripts.wam.tracking.preview.reset_multitask_live_requests import TASKS, make_task_request


def serve_adaptive_requests(
    *,
    registration,
    channel,
    image_root,
    directory,
    template,
    checkpoint,
    service,
    torch,
    counter,
    overrides,
    domain_reader,
    mask_evidence,
    result,
    response_gate=None,
    request_builder=make_task_request,
    response_validator=_check_response_identity,
):
    """Serve current-only requests until the registered limit or an explicit stop.

    The parent owns model initialization, source/checkpoint admission, CUDA
    binding, timeouts and process reaping. Attempts and partial denoise work are
    retained even if inference, validation or transport fails.
    """
    image_root, directory = Path(image_root).resolve(), Path(directory).resolve()
    prefix.require(
        type(registration) is dict
        and registration.get("task") in TASKS
        and type(registration.get("seed")) is int
        and registration["seed"] == 42
        and type(registration.get("denoise_steps")) is int
        and registration["denoise_steps"] == 35
        and type(registration.get("max_periods")) is int
        and type(registration.get("initial")) is dict
        and type(registration["initial"].get("step")) is int
        and registration["initial"]["step"] + registration["max_periods"] <= 600
        and image_root.is_relative_to(prefix.REPO / "outputs")
        and directory.is_relative_to(prefix.REPO / "outputs")
        and image_root.is_dir()
        and directory.is_dir()
        and registration.get("image_root") == str(image_root.relative_to(prefix.REPO))
        and not counter.calls
        and not mask_evidence
        and (response_gate is None or callable(response_gate))
        and callable(request_builder)
        and callable(response_validator),
        "A fresh registered task, sampling, paths and bounded period budget are required",
    )
    model_admission = AdaptiveModelAdmission(**registration)
    result.update(
        local_requests_received=0,
        model_requests_attempted=0,
        model_requests_completed=0,
        response_send_attempts=0,
        responses_sent=0,
        responses=[],
        model_sequence_stop=None,
        model_sequence_complete=False,
    )
    try:
        for index in range(registration["max_requests"]):
            request = channel.receive()
            if type(request) is dict and request.get("kind") == "stop":
                result["model_sequence_stop"] = check_adaptive_stop(model_admission, request)
                break
            result["local_requests_received"] += 1
            visible = model_admission.admit(request)
            prefix.require(
                set(visible) == {"image_bytes", "task"} and visible["task"] == registration["task"],
                "Only current image bytes and the public task are visible",
            )
            item = directory / f"step{request['step']}"
            item.mkdir()
            image_path = item / "current.png"
            with image_path.open("xb") as stream:
                stream.write(visible["image_bytes"])
            prefix.require(repro.sha256(image_path) == request["image"]["sha256"], "Copied current image differs")
            image = {
                "step": request["step"],
                "path": str(image_path.relative_to(prefix.REPO)),
                "sha256": request["image"]["sha256"],
            }
            model_request = request_builder(template, image, registration["task"])
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
                and caption["task_title"] == caption["command"] == registration["task"]
                and caption["task_plan"] == [],
                "Actual task conditioning differs",
            )
            repro.write_new(item / "conditioning.json", {"caption": caption, "image_sha256": image["sha256"]})
            result["model_requests_attempted"] += 1
            with torch.inference_mode(), counter.scope(model_request["name"], 35):
                response = service.execute(sample, batch).model_dump(mode="json")
                result["model_requests_completed"] += 1
            repro.write_new(item / "response.json", response)
            row = dict(
                step=request["step"],
                actual_response_ref=prefix.reference(item / "response.json"),
                model_request_ref=prefix.reference(item / "request.json"),
                model_elapsed_seconds=time.monotonic() - started,
                reference_sent=False,
            )
            result["responses"].append(row)
            arrays = repro.response_arrays(response)
            response_validator(response, model_request=model_request, task=registration["task"], checkpoint=checkpoint)
            prefix.require(
                repro.sha256(image_path) == request["image"]["sha256"]
                and model_inputs(request, image_root=registration["image_root"]) == visible
                and len(counter.calls) == 35 * (index + 1)
                and all(call["completed"] for call in counter.calls),
                "Every actual image and denoise call must remain valid",
            )
            engine.validate_mask_evidence(mask_evidence, index + 1)
            keys = ("version", "session", "request_id", "step", "sim_time_s", "state_sha256", "seed", "model_sha256")
            envelope = {
                **{key: request[key] for key in keys},
                "image_sha256": image["sha256"],
                "denoise_calls": 35,
                "action_dim_mask": [True] * 10 + [False] * 118,
                "action": arrays["action"].tolist(),
                "normalized_action": arrays["normalized_action"].tolist(),
            }
            if response_gate is not None:
                prefix.require(
                    response_gate(copy.deepcopy(request), copy.deepcopy(envelope)) is True,
                    "The registered current-response gate rejected this reference",
                )
            model_admission.complete(request_id=index, denoise_calls=35)
            result["response_send_attempts"] += 1
            channel.send(envelope)
            result["responses_sent"] += 1
            row["reference_sent"] = True
        if result["model_sequence_stop"] is None:
            prefix.require(
                model_admission.phase == "done" and result["responses_sent"] == registration["max_requests"],
                "All registered responses must complete",
            )
            result["model_sequence_complete"] = True
        else:
            prefix.require(
                model_admission.phase == "ready" and result["responses_sent"] == model_admission.admitted_requests,
                "A bounded stop must follow completed responses",
            )
    finally:
        result["denoise_calls_attempted"] = len(counter.calls)
        result["denoise_calls_completed"] = sum(bool(call["completed"]) for call in counter.calls)
        result["denoise_evidence"] = counter.calls
        result["mask_evidence"] = mask_evidence
