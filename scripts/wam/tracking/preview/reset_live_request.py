"""Serve one current-image reset continuation request through a WAM service."""

import copy
import json
import random
import time
from pathlib import Path

import numpy as np

from s2a.research.wam.tracking.runtime.causal_stream_control import check_stop
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalModelAdmission, model_inputs
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine
from scripts.wam.tracking.preview.live_stream_requests import _check_response_identity


def serve_one(
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
    request_builder=engine.make_request,
    response_validator=_check_response_identity,
):
    """Respond to one source-bound current image, never to a future observation."""
    prefix.require(
        registration["max_requests"] == 1 and registration["denoise_steps"] == 35, "One H40 request is registered"
    )
    model_admission = CausalModelAdmission(**registration)
    result.update(
        local_requests_received=0,
        model_requests_attempted=0,
        model_requests_completed=0,
        responses_sent=0,
        responses=[],
        model_sequence_stop=None,
    )
    try:
        request = channel.receive()
        if type(request) is dict and request.get("kind") == "stop":
            result["model_sequence_stop"] = check_stop(model_admission, request)
            return
        result["local_requests_received"] += 1
        prefix.require(
            request.get("state_sha256") == registration["initial"]["state_sha256"]
            and request.get("step") == registration["initial"]["step"]
            and request.get("sim_time_s") == registration["initial"]["sim_time_s"],
            "The request must bind the registered actual step-8 endpoint",
        )
        visible = model_admission.admit(request)
        prefix.require(
            set(visible) == {"image_bytes", "task"} and visible["task"] == registration["task"],
            "Only the current image and public task are visible",
        )
        item = Path(directory) / f"step{request['step']}"
        item.mkdir()
        image_path = item / "current.png"
        with image_path.open("xb") as stream:
            stream.write(visible["image_bytes"])
        prefix.require(repro.sha256(image_path) == request["image"]["sha256"], "Copied image differs")
        image = {
            "step": request["step"],
            "path": str(image_path.relative_to(prefix.REPO)),
            "sha256": request["image"]["sha256"],
        }
        model_request = request_builder(template, image)
        repro.write_new(item / "request.json", model_request)
        random.seed(registration["seed"])
        np.random.seed(registration["seed"])
        torch.manual_seed(registration["seed"])
        torch.cuda.manual_seed_all(registration["seed"])
        started = time.monotonic()
        sample, batch = service.prepare(overrides.model_validate(copy.deepcopy(model_request)), item)
        domain = domain_reader(service, sample, batch)
        caption = json.loads(batch["ai_caption"][0])
        prefix.require(
            domain.is_libero
            and caption["task_title"] == caption["command"] == registration["task"]
            and caption["task_plan"] == [],
            "Actual WAM conditioning differs",
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
            and model_inputs(request, image_root=str(image_root.relative_to(prefix.REPO))) == visible
            and len(counter.calls) == 35
            and all(call["completed"] for call in counter.calls),
            "Image and actual denoise sequence must remain valid",
        )
        engine.validate_mask_evidence(mask_evidence, 1)
        model_admission.complete(request_id=0, denoise_calls=35)
        keys = ("version", "session", "request_id", "step", "sim_time_s", "state_sha256", "seed", "model_sha256")
        envelope = {
            **{key: request[key] for key in keys},
            "image_sha256": image["sha256"],
            "denoise_calls": 35,
            "action_dim_mask": [True] * 10 + [False] * 118,
            "action": arrays["action"].tolist(),
            "normalized_action": arrays["normalized_action"].tolist(),
        }
        channel.send(envelope)
        result["responses_sent"] += 1
        row["reference_sent"] = True
        prefix.require(
            model_admission.phase == "done" and result["responses_sent"] == 1,
            "One response must close the registered request",
        )
    finally:
        result["denoise_calls_attempted"] = len(counter.calls)
        result["denoise_calls_completed"] = sum(bool(call["completed"]) for call in counter.calls)
        result["denoise_evidence"] = counter.calls
        result["mask_evidence"] = mask_evidence
