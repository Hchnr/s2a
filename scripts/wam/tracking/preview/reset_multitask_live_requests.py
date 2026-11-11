"""Serve bounded current-image WAM requests for a registered development task."""

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

TASKS = (
    "put the bowl on the plate",
    "put the wine bottle on the rack",
    "put the cream cheese in the bowl",
)


def make_task_request(template, image, task):
    """Preserve the registered Action128 sampling while substituting current inputs."""
    prefix.require(
        task in TASKS and type(image.get("step")) is int and image["step"] >= 0,
        "One registered task and current step required",
    )
    request = copy.deepcopy(template)
    request["name"] = f"s2a_reset_task{TASKS.index(task)}_t{image['step']}_s42"
    request["prompt"] = task
    request["vision_path"] = str(prefix.REPO / image["path"])
    request["extra"]["action128"]["task_title"] = task
    normalized = copy.deepcopy(request)
    normalized["name"] = "s2a_task0_init0_s42_0"
    normalized["prompt"] = normalized["extra"]["action128"]["task_title"] = TASKS[0]
    repro.validate_request(normalized, {"path": request["vision_path"]})
    extra = request["extra"]["action128"]
    prefix.require(
        extra["selected_roles"] == ["auxiliary_0", "left_wrist"]
        and extra["camera_layout"] == [["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
        "The original dual-camera conditioning layout is required",
    )
    return request


def serve_live_requests(
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
    request_builder=make_task_request,
    response_validator=_check_response_identity,
):
    """Serve up to five consecutive real current-image requests or a bounded stop."""
    prefix.require(
        registration["task"] in TASKS and registration["max_requests"] == 5 and registration["denoise_steps"] == 35,
        "Five bounded requests for one registered task are required",
    )
    model_admission = CausalModelAdmission(**registration)
    result.update(
        local_requests_received=0,
        model_requests_attempted=0,
        model_requests_completed=0,
        responses_sent=0,
        responses=[],
        model_sequence_stop=None,
        model_sequence_complete=False,
    )
    try:
        for index in range(registration["max_requests"]):
            request = channel.receive()
            if type(request) is dict and request.get("kind") == "stop":
                result["model_sequence_stop"] = check_stop(model_admission, request)
                break
            result["local_requests_received"] += 1
            if index == 0:
                prefix.require(
                    all(
                        request.get(key) == registration["initial"][key]
                        for key in ("step", "sim_time_s", "state_sha256")
                    ),
                    "The first request must bind the actual registered endpoint",
                )
            visible = model_admission.admit(request)
            prefix.require(
                set(visible) == {"image_bytes", "task"} and visible["task"] == registration["task"],
                "Only current image bytes and the public task are visible",
            )
            item = Path(directory) / f"step{request['step']}"
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
                and model_inputs(request, image_root=str(image_root.relative_to(prefix.REPO))) == visible
                and len(counter.calls) == 35 * (index + 1)
                and all(call["completed"] for call in counter.calls),
                "Every actual image and denoise call must remain valid",
            )
            engine.validate_mask_evidence(mask_evidence, index + 1)
            model_admission.complete(request_id=index, denoise_calls=35)
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
