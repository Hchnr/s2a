"""Decode and bind actual model exchanges for independent physical replay."""

from pathlib import Path

import numpy as np

from s2a.geometry import validate_transform
from s2a.references.action_reference import decode_action_targets
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import causal_stream_recorded_admission as historical
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical


def decode_actual_references(inputs, branch, *, image_root, max_windows):
    """Validate each observed exchange, then decode its saved Action128 output.

    The replay caller separately checks the captured pose against its own
    physical endpoint before using each target. No policy or model is loaded.
    """
    runtime = branch["runtime"]
    stream = runtime["continuation"]
    segments, captures, exchanges = (
        stream["segments"],
        runtime["streaming_captures"],
        runtime["streaming_model_exchanges"],
    )
    prefix.require(
        type(max_windows) is int
        and max_windows in (2, 6)
        and len(segments) == len(captures) == len(exchanges) <= max_windows,
        "Every actual stream segment requires one current capture and model exchange",
    )
    registration = physical.model_config(inputs, image_root)
    registration["max_requests"] = max_windows
    ledger = CausalStreamLedger(**registration)
    tool = validate_transform(np.asarray(inputs["old"]["old"]["base"]["previous"]["tool"], dtype=float))
    chunks, sources = [], []
    for index, (segment, capture, exchange) in enumerate(zip(segments, captures, exchanges, strict=True)):
        current = ledger.current
        binding = {"sim_time": current["sim_time_s"], "sim_state_sha256": current["state_sha256"]}
        endpoint = {"endpoint_step": current["step"], **binding}
        prefix.require(
            segment["start_step"] == capture["frame_index"] == exchange["step"] == current["step"]
            and exchange["request_id"] == index
            and all(
                exchange.get(key) is True
                for key in ("request_sent", "response_received", "response_accepted", "policy_ready")
            )
            and capture["current_endpoint"] == endpoint
            and capture["render_before"] == capture["render_after"] == binding
            and capture["future_rgb_frames"] == 0,
            "Actual image, state, time and exchange identities must agree",
        )
        request = prefix.read(exchange["request_ref"])
        response = prefix.read(exchange["response_ref"])
        expected_image = Path(image_root) / request["image"]["path"]
        prefix.require(
            capture["canvas_png"]["path"] == str(expected_image)
            and capture["canvas_png"]["sha256"] == request["image"]["sha256"],
            "The model request must bind its actual current capture",
        )
        issued = ledger.issue(before_render=current, after_render=current, image=request["image"])
        prefix.require(issued == request, "Saved request differs from the independently sequenced ledger")
        action = ledger.accept(response, after_wait=current)
        if index < 2:
            historical.gate_response(inputs["records"], request, response)
        source_pose = validate_transform(np.asarray(capture["current_source_pose"], dtype=float))
        opening = capture["current_opening"]
        prefix.require(
            source_pose.shape == (4, 4)
            and type(opening) in (int, float)
            and np.isfinite(opening)
            and 0 <= opening <= 1,
            "One finite captured source pose and opening are required",
        )
        poses, _openings = decode_action_targets(source_pose @ tool, action, horizon=40)
        chunks.append(poses[:8])
        sources.append({"source_pose": source_pose, "opening": opening})
        for period in segment["continuation"]["periods"]:
            before = ledger.current
            end = period["endpoint_receipt"]
            after = {
                "step": end["endpoint_step"],
                "sim_time_s": end["sim_time"],
                "state_sha256": end["sim_state_sha256"],
            }
            prefix.require(
                period["step"] == before["step"] and state_sha256(period["actual_state"]) == after["state_sha256"],
                "Every reference boundary follows an actually recorded physical endpoint",
            )
            ledger.complete_period(before=before, after=after, inner_samples=len(period["actual_inner"]))
    prefix.require(
        ledger.issued_requests == stream["requests_issued"]
        and ledger.accepted_requests == stream["responses_accepted"]
        and ledger.completed_periods == stream["feedback_periods_committed"]
        and (
            (stream["complete"] is True and ledger.phase == "done" and stream["stop"] is None)
            or (
                stream["complete"] is False
                and type(stream["stop"]) is dict
                and stream["stop"].get("at_step") == ledger.current["step"]
                and stream["stop"].get("next_action_dispatched") is False
            )
        ),
        "Actual exchanges and completed periods must close or stop explicitly within budget",
    )
    return chunks, sources
