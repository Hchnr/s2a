"""Summarize reset-origin adaptive runs and compare independent replays."""

from pathlib import Path

import numpy as np

from scripts.simulation.runtime import adaptive_live_bowl_recovery_audit as costs
from scripts.simulation.runtime import adaptive_live_bowl_worker as historical
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as capture_check
from scripts.wam.tracking.preview import adaptive_multiseed_contract as contract
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine

add_cost_bounds = costs.add_cost_bounds
exact_cost_bounds = costs.exact_cost_bounds


def periods(branch):
    sequence = branch["runtime"]["continuation"]
    first = sequence["first_window"]["periods"]
    adaptive = [row for segment in sequence["adaptive"]["segments"] for row in segment["continuation"]["periods"]]
    return first, adaptive


def task_outcomes(flags):
    """Score the preregistered full-length task criterion without filling gaps."""
    prefix.require(type(flags) is list and flags and all(type(flag) is bool for flag in flags), "Native task flags")
    tail = 0
    for flag in reversed(flags):
        if not flag:
            break
        tail += 1
    return dict(
        task_success_periods=sum(flags),
        final_success=flags[-1],
        consecutive_success_tail=tail,
        stable_last40=len(flags) == 600 and tail >= 40,
    )


def summarize_branch(branch, phase, config):
    """Count actual physical work, task results and the exact recorded stop."""
    prefix.require(phase in ("producer", "checker"), "A producer or checker is required")
    runtime, sequence = branch["runtime"], branch["runtime"]["continuation"]
    first, adaptive = periods(branch)
    rows, counters, stream = first + adaptive, runtime["counters"], sequence["adaptive"]
    count = len(rows)
    prefix.require(
        branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and runtime["cleanup_complete"] is runtime["reset_observation_verified"] is runtime["reset_image_match"] is True
        and runtime["previous_saved_command"] is None
        and branch["case"]["case_id"] == config["case_id"]
        and len(first) == config["prefix_steps"] == 8
        and len(adaptive) <= config["max_adaptive_periods"]
        and [row["step"] for row in rows] == list(range(count))
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == count
        and counters["mj_step"]["attempted"] == counters["mj_step"]["completed"] == 25 * count
        and counters["model_construct"]["completed"] == config["max_models_per_phase"]
        and counters["context"]["completed"] == config["max_contexts_per_phase"]
        and all(len(row["actual_inner"]) == 25 for row in rows),
        "Reset, actual dispatches and physical budgets must agree",
    )
    if phase == "producer":
        stop = stream["stop"]
        prefix.require(
            runtime["newly_selected_commands"] == count
            and stream["control_steps_attempted"]
            == stream["control_steps_completed"]
            == stream["feedback_periods_committed"]
            == len(adaptive)
            and stream["requests_issued"] == stream["responses_accepted"] == len(stream["segments"])
            and all(row["inflight"] is None for row in stream["segments"]),
            "Every selected action and model exchange must be accounted for",
        )
        frame = runtime["adaptive_stop_frame"]
        if stream["requests_issued"] < config["max_model_requests"]:
            prefix.require(
                frame.get("required") is frame.get("send_attempted") is frame.get("sent") is True
                and frame.get("frame")
                == {"version": 1, "kind": "stop", "session": config["session"], "next_step": count},
                "A waiting model must receive the exact last completed step",
            )
        else:
            prefix.require(
                frame == {"required": False, "reason": "model_request_budget_exhausted"},
                "The exhausted model budget cannot receive an extra request or stop",
            )
        receipts = runtime["segment_receipts"]
        budget = dict(
            quality_responses=sum(row["quality_refiner"]["quality_response_evaluations"] for row in receipts),
            baseline_responses=sum(row["quality_refiner"]["baseline_response_evaluations"] for row in receipts),
            gripper_sequences=sum(row["gripper_proposals"]["candidate_sequences"] for row in receipts),
            endpoint_queries=sum(row["quality_refiner"]["endpoint_queries_charged"] for row in receipts),
            transition_steps=sum(row["quality_refiner"]["transition_steps_charged"] for row in receipts),
        )
        maximum = config["max_control_steps_per_phase"]
        prefix.require(
            all(type(value) is int and value >= 0 for value in budget.values())
            and budget["quality_responses"] <= maximum * 320
            and budget["baseline_responses"] <= maximum * 480
            and budget["gripper_sequences"] <= maximum * 6561
            and budget["endpoint_queries"] <= maximum * 200
            and budget["transition_steps"] <= maximum * 2600,
            "Candidate work must stay within the original per-period bounds",
        )
    else:
        stop, budget = stream["reported_producer_stop"], None
        prefix.require(
            branch["candidate_modules_unloaded"] is True
            and runtime["newly_selected_commands"] == 0
            and stream["complete"] is True
            and stream["stop_reason_independently_verified"] is False
            and stream["control_steps_attempted"] == stream["control_steps_completed"] == len(adaptive),
            "The checker must replay only recorded actions without recomputing policy stops",
        )
    prefix.require(
        type(stop) is dict
        and type(stop.get("reason")) is str
        and bool(stop["reason"])
        and stop.get("next_action_dispatched") is False
        and (
            (type(stop.get("at_step")) is int and "after_step" not in stop and stop["at_step"] == count)
            or (type(stop.get("after_step")) is int and "at_step" not in stop and stop["after_step"] + 1 == count)
        ),
        "The stop must immediately follow the actually completed actions",
    )
    buffer_losses = [row["step"] for row in rows if row["inner_below_buffer_samples"]]
    outside = [row["step"] for row in rows if row["inner_outside_samples"]]
    prefix.require(
        (
            not buffer_losses
            or (
                buffer_losses == [count - 1]
                and stop["reason"] in ("observed_joint_buffer_loss", "actual_inner_joint_limit_violation")
            )
        )
        and (not outside or (outside == [count - 1] and stop["reason"] == "actual_inner_joint_limit_violation")),
        "No action may follow a measured joint-buffer loss or limit violation",
    )
    errors = np.asarray([row["tcp_error"] for row in rows], dtype=float)
    prefix.require(errors.shape == (count, 6) and np.isfinite(errors).all(), "Finite measured TCP errors are required")
    task = task_outcomes([row["task_success"] for row in rows])
    return dict(
        control_steps=count,
        native_steps=25 * count,
        models=config["max_models_per_phase"],
        contexts=config["max_contexts_per_phase"],
        prefix_steps=len(first),
        adaptive_steps=len(adaptive),
        model_requests=stream["requests_issued"] if phase == "producer" else 0,
        stop=stop,
        candidate_budget=budget,
        pose_caps_satisfied=sum(bool(row["pose_caps_satisfied"]) for row in rows),
        position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
        minimum_inner_margin_rad=float(min(row["minimum_inner_margin_rad"] for row in rows)),
        target_grasp_periods=sum(bool(row["target_grasp"]) for row in rows),
        **task,
    )


def compare_branches(producer, checker):
    """Compare all dispatched periods and independently re-rendered images."""
    first, adaptive = periods(producer)
    replay_first, replay_adaptive = periods(checker)
    difference = comparison.compare_periods(first + adaptive, replay_first + replay_adaptive)
    expected, actual = producer["runtime"]["adaptive_captures"], checker["runtime"]["adaptive_checker_captures"]
    prefix.require(len(expected) == len(actual), "Every request image must be independently reproduced")
    for left, right in zip(expected, actual, strict=True):
        capture_check.compare_capture(left, right)
    return difference


def verify_model_exchange(branch, model, bundle):
    """Bind every accepted reference to the actual seed, model and current image."""
    registration = bundle["model_registration"]
    responses = historical.load_recorded_responses(branch, model)
    prefix.require(
        model["sampling_seed"] == registration["seed"]
        and model["local_requests_received"] == model["responses_sent"] == len(responses)
        and len(model["denoise_evidence"]) == 35 * len(responses)
        and all(row["completed"] is True for row in model["denoise_evidence"]),
        "Every accepted reference must have the registered seed and actual denoising evidence",
    )
    frame = branch["runtime"]["adaptive_stop_frame"]
    if len(responses) < registration["max_requests"]:
        prefix.require(
            model["model_sequence_complete"] is False
            and model["model_sequence_stop"] == frame["frame"]
            and frame["sent"] is True,
            "The model must receive the producer's exact bounded stop",
        )
    else:
        prefix.require(
            model["model_sequence_complete"] is True and model["model_sequence_stop"] is None,
            "The exhausted request budget must close the complete model sequence",
        )
    engine.validate_mask_evidence(model["mask_evidence"], len(responses))
    checkpoint = bundle["prepared"]["model_config"]["checkpoint"]
    for row, capture in zip(model["responses"], branch["runtime"]["adaptive_captures"], strict=True):
        request, response = prefix.read(row["model_request_ref"]), prefix.read(row["actual_response_ref"])
        contract.validate_response(
            response, model_request=request, task=registration["task"], checkpoint=checkpoint, seed=registration["seed"]
        )
        image_path = Path(request["vision_path"])
        prefix.require(
            image_path == Path(row["model_request_ref"]["path"]).parent / "current.png",
            "The model request must use its own recorded current-image copy",
        )
        prefix.checked({"path": str(image_path), "sha256": capture["canvas_png"]["sha256"]})
    return dict(requests=len(responses), denoise_calls=35 * len(responses))
