"""Replay one saved development image to the registered model, without physics."""

import os
import socket
from pathlib import Path

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveStreamLedger
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro


def exchange_saved_frame(inputs, *, registration, image_root, channel):
    """Exchange the recorded image and compare arrays with the old seed-42 output.

    The endpoint snapshot is a historical identity in this replay. No current
    simulator exists here and no action is selected or dispatched.
    """
    image_root = Path(image_root).resolve()
    config = inputs["config"]
    prefix.require(
        image_root.is_relative_to(prefix.REPO / "outputs")
        and image_root.is_dir()
        and registration["image_root"] == str(image_root.relative_to(prefix.REPO))
        and registration["initial"] == inputs["initial"]
        and registration["seed"] == config["seed"] == 43
        and registration["task"] == config["task"]
        and registration["model_sha256"] == config["model_sha256"]
        and registration["max_requests"] == registration["max_periods"] == 1
        and registration["denoise_steps"] == 35,
        "The single registered saved-frame model exchange is required",
    )
    prefix.checked(inputs["refs"]["historical_image"])
    image_path = image_root / "current.png"
    prefix.require(repro.sha256(image_path) == config["image_sha256"], "Copied historical image differs")
    ledger = AdaptiveStreamLedger(**registration)
    recorded = dict(inputs["initial"])
    request = ledger.issue(
        before_render=recorded,
        after_render=recorded,
        image=dict(path="current.png", sha256=config["image_sha256"]),
    )
    channel.send(request)
    response = channel.receive()
    ledger.accept(response, after_wait=recorded)
    prefix.checked(inputs["refs"]["historical_image"])
    prefix.require(repro.sha256(image_path) == config["image_sha256"], "Replay image changed during the request")
    comparable = {
        "status": "success",
        "outputs": [
            {
                "content": {
                    "action": response["action"],
                    "action128": {"normalized_action": response["normalized_action"]},
                }
            }
        ],
    }
    comparison = repro.compare_responses(comparable, inputs["previous_response"])
    return dict(
        request=request,
        response=response,
        compared_to_saved_seed42=comparison,
        snapshot_scope="historical_recorded_endpoint_not_live",
        accepted_requests=ledger.accepted_requests,
        completed_periods=ledger.completed_periods,
        physics_steps=0,
        native_models=0,
        action_dispatches=0,
    )


def replay_worker(*, output, loader):
    """Publish partial or complete offline-client evidence and close the socket."""
    prefix.environment()
    output = Path(output).resolve()
    directory = output / "producer/simulation"
    prefix.require(
        output.is_relative_to(prefix.REPO / "outputs") and directory.is_dir(), "Local replay output required"
    )
    prefix.require(callable(loader), "A registered replay input loader is required")
    result = dict(
        complete=False,
        passed=False,
        evidence_level="saved_development_image_model_replay",
        physics_steps=0,
        native_models=0,
        action_dispatches=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    channel = None
    try:
        inputs, registration = loader()
        channel = FramedJSONChannel(
            socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])), max_bytes=2 * 1024 * 1024, timeout_seconds=1750
        )
        result["exchange"] = exchange_saved_frame(
            inputs, registration=registration, image_root=directory, channel=channel
        )
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if channel is not None:
            channel.close()
        print(publish(directory / "result.json", result), flush=True)
