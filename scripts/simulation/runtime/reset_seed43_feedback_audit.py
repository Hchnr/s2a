"""Summarize and compare the reset/current-image physical feedback receipts."""

import numpy as np

from s2a.research.wam.tracking.runtime.causal_stream_control import make_stop
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as capture
from scripts.simulation.runtime import run_reset_live_second_window as previous
from scripts.wam.tracking.preview import adaptive_multiseed_contract as contract
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as inference


def periods(branch, phase):
    """Return the completed first and live periods, including early-stop prefixes."""
    sequence = branch["runtime"]["continuation"]
    first = sequence["first_window"]["periods"]
    if phase == "producer":
        segments = sequence["stream"]["segments"]
        prefix.require(len(segments) <= 1, "At most one model reference is registered")
        live = [] if not segments else segments[0]["continuation"]["periods"]
    else:
        prefix.require(phase == "checker", "A producer or independent checker is required")
        replay = sequence["replay"]
        live = [] if replay is None else replay["periods"]
    return first, live


def summarize_branch(branch, phase, config, *, producer=None):
    """Check original physical budgets and report measured task outcomes."""
    prefix.require(
        config["first_window_steps"] == config["live_window_steps"] == 8
        and config["max_control_steps_per_phase"] == 16
        and config["max_native_steps_per_phase"] == 400
        and config["max_models_per_phase"] == config["max_contexts_per_phase"] == 3
        and branch["case"]["case_id"] == config["case_id"],
        "The exact registered case and two-window physical bounds are required",
    )
    counts = previous.summarize_branch(branch, phase, producer=producer)
    first, live = periods(branch, phase)
    rows = first + live
    error = np.asarray([row["tcp_error"] for row in rows], dtype=float)
    prefix.require(
        len(rows) == counts["control_steps"]
        and len(live) == counts["live_control_steps"]
        and error.shape == (len(rows), 6)
        and np.isfinite(error).all(),
        "Finite measured TCP errors are required for every completed period",
    )
    counts.update(
        pose_caps_satisfied=sum(bool(row["pose_caps_satisfied"]) for row in rows),
        position_rmse_m=float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1)))),
        minimum_inner_margin_rad=float(min(row["minimum_inner_margin_rad"] for row in rows)),
        target_grasp_periods=sum(bool(row["target_grasp"]) for row in rows),
        task_success_periods=sum(bool(row["task_success"]) for row in rows),
        live_target_grasp_periods=sum(bool(row["target_grasp"]) for row in live),
        live_task_success_periods=sum(bool(row["task_success"]) for row in live),
    )
    return counts


def compare_branches(producer, checker):
    """Compare physical samples, task labels and each actual current image."""
    first, live = periods(producer, "producer")
    replay_first, replay_live = periods(checker, "checker")
    difference = comparison.compare_periods(first + live, replay_first + replay_live)
    expected = producer["runtime"]["streaming_captures"]
    actual = checker["runtime"].get("checker_stream_capture")
    prefix.require(len(expected) <= 1, "At most one independent current image is registered")
    if expected:
        prefix.require(type(actual) is dict, "The independent current-image capture is required")
        capture.compare_capture(expected[0], actual)
    else:
        prefix.require(actual is None, "A pre-request stop has no independent current image")
    prefix.require(
        checker["runtime"]["continuation"]["reported_producer_stop"]
        == producer["runtime"]["continuation"]["stream"]["stop"]
        and checker["runtime"]["continuation"]["stop_reason_independently_verified"] is False,
        "Replay must retain the producer stop without claiming to recompute its reason",
    )
    return difference


def verify_model_exchange(producer, model, bundle, tool):
    """Bind the dispatched reference to the actual current image and model output."""
    registration = bundle["model_registration"]
    runtime = producer["runtime"]
    stream = runtime["continuation"]["stream"]
    captures, exchanges, rows = runtime["streaming_captures"], runtime["streaming_model_exchanges"], model["responses"]
    count = model["responses_sent"]
    prefix.require(
        type(count) is int
        and count in (0, 1)
        and len(captures) == len(exchanges) == len(rows) == count
        and model["local_requests_received"]
        == model["model_requests_attempted"]
        == model["model_requests_completed"]
        == stream["requests_issued"]
        == stream["responses_accepted"]
        == count
        and model["denoise_calls_attempted"] == model["denoise_calls_completed"] == 35 * count
        and len(model["denoise_evidence"]) == 35 * count
        and all(row["completed"] is True for row in model["denoise_evidence"]),
        "Each accepted reference must have one completed model response and 35 denoising calls",
    )
    inference.validate_mask_evidence(model["mask_evidence"], count)
    if not count:
        prefix.require(
            stream["stop"] is not None
            and stream["stop"]["next_action_dispatched"] is False
            and runtime["model_stop_sent"] is True
            and model["model_sequence_complete"] is False
            and model["model_sequence_stop"] == make_stop(session=registration["session"], next_step=8),
            "An empty model exchange requires the exact pre-request stop",
        )
        return {"requests": 0, "denoise_calls": 0}
    row, observed, exchange = rows[0], captures[0], exchanges[0]
    prefix.require(
        row["step"] == observed["frame_index"] == exchange["step"] == 8
        and row["reference_sent"] is True
        and runtime["model_stop_sent"] is False
        and model["model_sequence_complete"] is True
        and model["model_sequence_stop"] is None,
        "The one response must be sent for the actual step-8 current image",
    )
    request = prefix.read(row["model_request_ref"])
    response = prefix.read(row["actual_response_ref"])
    contract.validate_response(
        response,
        model_request=request,
        task=registration["task"],
        checkpoint=bundle["prepared"]["model_config"]["checkpoint"],
        seed=registration["seed"],
    )
    capture.decode_response(registration, observed, exchange, response, tool)
    return {"requests": 1, "denoise_calls": 35}
