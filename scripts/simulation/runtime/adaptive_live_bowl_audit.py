"""Audit actual producer and independent checker receipts for an adaptive trial."""

import numpy as np

from scripts.simulation.runtime import adaptive_live_bowl_admission as admission
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import causal_replan_step107_execution as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_two_window_replay as capture_check


def periods(branch, phase):
    sequence = branch["runtime"]["continuation"]
    prior = previous.prefix_periods(sequence["prefix107"], replay=phase == "checker")
    adaptive = [row for segment in sequence["adaptive"]["segments"] for row in segment["continuation"]["periods"]]
    return prior, adaptive


def summarize_branch(branch, phase, config):
    """Count only completed physical actions, actual inner steps and measured labels."""
    prefix.require(phase in ("producer", "checker"), "A registered physical phase is required")
    runtime, sequence = branch["runtime"], branch["runtime"]["continuation"]
    prior, adaptive = periods(branch, phase)
    rows, counters, stream = prior + adaptive, runtime["counters"], sequence["adaptive"]
    count = len(rows)
    prefix.require(
        branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
        and runtime["cleanup_complete"] is runtime["reset_observation_verified"] is runtime["reset_image_match"] is True
        and runtime["previous_saved_command"] is None
        and branch["case"]["case_id"] == config["case_id"]
        and len(prior) == config["prefix_steps"]
        and len(adaptive) <= config["max_adaptive_periods"]
        and [row["step"] for row in rows] == list(range(count))
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == count
        and counters["mj_step"]["attempted"] == counters["mj_step"]["completed"] == 25 * count
        and counters["model_construct"]["completed"] == config["max_models_per_phase"]
        and counters["context"]["completed"] == config["max_contexts_per_phase"]
        and all(len(row["actual_inner"]) == 25 for row in rows),
        "Actual commands, endpoints, simulation models and physical budgets must agree",
    )
    first_stop = sequence["prefix107"]["new_reference"]["stop"]
    prefix.require(
        first_stop["reason"] == "requested_observation"
        and first_stop["at_step"] == config["prefix_steps"]
        and first_stop["next_action_dispatched"] is False,
        "The adaptive sequence must start at the registered completed prefix",
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
            "Every producer dispatch and reference must be fully accounted for",
        )
        frame = runtime["adaptive_stop_frame"]
        if stream["requests_issued"] < config["max_model_requests"]:
            prefix.require(
                frame.get("required") is frame.get("send_attempted") is frame.get("sent") is True
                and frame.get("frame")
                == {"version": 1, "kind": "stop", "session": admission.SESSION, "next_step": count},
                "A waiting model must receive a stop after the actual last completed endpoint",
            )
        else:
            prefix.require(
                frame == {"required": False, "reason": "model_request_budget_exhausted"},
                "A fully served model request budget must not receive an extra stop",
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
            "Candidate work must remain within the original per-period limits",
        )
    else:
        stop = stream["reported_producer_stop"]
        budget = None
        prefix.require(
            branch["candidate_modules_unloaded"] is True
            and runtime["newly_selected_commands"] == 0
            and stream["complete"] is True
            and stream["stop_reason_independently_verified"] is False
            and stream["control_steps_attempted"] == stream["control_steps_completed"] == len(adaptive),
            "The isolated checker must replay only completed actions without recomputing policy stops",
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
        "The recorded stop must immediately follow the actual completed actions",
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
        "No action may follow a measured buffer loss or joint-limit violation",
    )
    errors = np.asarray([row["tcp_error"] for row in rows], dtype=float)
    prefix.require(errors.shape == (count, 6) and np.isfinite(errors).all(), "Finite measured TCP errors required")
    return dict(
        control_steps=count,
        native_steps=25 * count,
        models=config["max_models_per_phase"],
        contexts=config["max_contexts_per_phase"],
        prefix_steps=len(prior),
        adaptive_steps=len(adaptive),
        stop=stop,
        candidate_budget=budget,
        pose_caps_satisfied=sum(bool(row["pose_caps_satisfied"]) for row in rows),
        position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
        minimum_inner_margin_rad=float(min(row["minimum_inner_margin_rad"] for row in rows)),
        target_grasp_periods=sum(bool(row["target_grasp"]) for row in rows),
        task_success_periods=sum(bool(row["task_success"]) for row in rows),
        adaptive_target_grasp_periods=sum(bool(row["target_grasp"]) for row in adaptive),
        adaptive_task_success_periods=sum(bool(row["task_success"]) for row in adaptive),
    )


def compare_branches(producer, checker):
    """Compare physical observations and re-rendered current images, not solver stops."""
    produced_prefix, produced_adaptive = periods(producer, "producer")
    checked_prefix, checked_adaptive = periods(checker, "checker")
    difference = comparison.compare_periods(produced_prefix + produced_adaptive, checked_prefix + checked_adaptive)
    expected = producer["runtime"]["adaptive_captures"]
    actual = checker["runtime"]["adaptive_checker_captures"]
    prefix.require(len(expected) == len(actual), "Every producer image must be independently reproduced")
    for left, right in zip(expected, actual, strict=True):
        capture_check.compare_capture(left, right)
    return difference
