"""Serve the registered reset-to-current-image seed-43 request."""

from pathlib import Path

from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalModelAdmission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_seed43_live_admission as admission
from scripts.wam.tracking.preview import adaptive_multiseed_contract as contract
from scripts.wam.tracking.preview import reset_live_request as legacy


def validate_registration(registration, output):
    """Reject changed seed, frame, checkpoint identity or model budget."""
    config = prefix.read({"path": str(admission.CONFIG), "sha256": admission.CONFIG_SHA})
    branch = prefix.read(admission.configured_ref(config, "historical_case_branch"))
    endpoint = branch["runtime"]["continuation"]["periods"][-1]["endpoint_receipt"]
    expected = dict(
        step=endpoint["endpoint_step"], sim_time_s=endpoint["sim_time"], state_sha256=endpoint["sim_state_sha256"]
    )
    output = Path(output).resolve()
    prefix.require(
        type(registration) is dict
        and output.is_relative_to(prefix.REPO / "outputs")
        and type(registration.get("initial")) is dict
        and admission._same(registration["initial"], expected)
        and type(registration["initial"].get("step")) is int
        and registration["initial"]["step"] == config["first_window_steps"] == 8
        and registration["initial"].get("state_sha256") == config["historical_step8_state_sha256"]
        and type(registration.get("seed")) is int
        and registration["seed"] == config["sampling_seed"] == 43
        and registration.get("task") == config["task"] == "put the bowl on the plate"
        and registration.get("session") == admission.SESSION
        and registration.get("model_sha256") == config["model_sha256"]
        and type(registration.get("max_requests")) is int
        and registration["max_requests"] == config["max_model_requests"] == 1
        and type(registration.get("denoise_steps")) is int
        and registration["denoise_steps"] == config["denoise_calls_per_request"] == 35
        and type(registration.get("image_root")) is str
        and (prefix.REPO / registration["image_root"]).resolve() == output / "producer/simulation/runtime",
        "The registered seed-43 current-image request and bounds are required",
    )
    return CausalModelAdmission(**registration)


def serve_seed43(
    *,
    output,
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
):
    """Use the legacy causal loop with the actual registered task and seed."""
    output, image_root, directory = Path(output).resolve(), Path(image_root).resolve(), Path(directory).resolve()
    validate_registration(registration, output)
    prefix.require(
        image_root == output / "producer/simulation/runtime"
        and directory.is_relative_to(output / "producer/model")
        and directory.is_dir()
        and not counter.calls
        and not mask_evidence,
        "Fresh model counters and the registered image and model directories are required",
    )

    def request_builder(request_template, image):
        request = contract.make_task_request(request_template, image, registration["task"], registration["seed"])
        prefix.require(
            type(request.get("seed")) is int
            and request["seed"] == 43
            and request["prompt"] == registration["task"]
            and request["vision_path"] == str(prefix.REPO / image["path"]),
            "The actual model request must use the registered seed, task and current image",
        )
        return request

    def response_validator(response, *, model_request, task, checkpoint):
        contract.validate_response(response, model_request=model_request, task=task, checkpoint=checkpoint, seed=43)

    result.update(sampling_seed=43, model_sequence_complete=False)
    legacy.serve_one(
        registration=registration,
        channel=channel,
        image_root=image_root,
        directory=directory,
        template=template,
        checkpoint=checkpoint,
        service=service,
        torch=torch,
        counter=counter,
        overrides=overrides,
        domain_reader=domain_reader,
        mask_evidence=mask_evidence,
        result=result,
        request_builder=request_builder,
        response_validator=response_validator,
    )
    result["model_sequence_complete"] = result["responses_sent"] == 1
