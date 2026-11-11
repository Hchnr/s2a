"""Bind a static stop-state study to the independently audited live stream."""

import numpy as np

from s2a.geometry import finite, validate_transform
from s2a.references.action_reference import decode_action_targets
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.reference_ports import audit_causal_stream_live_cached as live_audit
from scripts.wam.tracking.preview import causal_stream_live_cached_admission as live

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/cached_live_stop125_feasibility_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/cached_live_stop125_feasibility_20261111.md"
OUTPUT = ROOT / "outputs/cached_live_stop125_feasibility_20261111"
QA = ROOT / "outputs/qa/cached_live_stop125_feasibility_20261111/result.json"
DRY = ROOT / "outputs/cached_live_stop125_feasibility_dry_20261111/result.json"

FIXED = {
    "schema": "s2a.cached-live-stop125-feasibility.v1",
    "output": "outputs/cached_live_stop125_feasibility_20261111",
    "cached_live_audit": "outputs/causal_stream_live_cached_audit_20261111/result.json",
    "cached_live_audit_sha256": "65f51ff8b9135ba01462ec96f662ed648024eb1df5e8dee1b3fef09dbaaa87c9",
    "case_index": 8,
    "seed": 42,
    "stop_step": 125,
    "generated_step": 120,
    "reference_cursor": 5,
    "branches": ["quality_past", "quality_nominal", "joint_past", "quality_past_warm"],
    "control_dt_s": 0.05,
    "inner_dt_s": 0.002,
    "joint_buffer_rad": 0.05,
    "braking_time_s": 0.1,
    "position_cap_m": 0.01,
    "rotation_cap_rad": 0.05,
    "arm_rate_limit": 0.25,
    "max_models": 1,
    "max_forward_queries": 1,
    "max_optimizer_calls": 4,
    "max_optimizer_responses": 640,
    "max_independent_endpoint_queries": 800,
    "max_independent_transition_steps": 10400,
    "physics_steps": 0,
    "world_model_calls": 0,
    "fresh_cases": 0,
    "worker_timeout_seconds": 1200,
    "automatic_retry": False,
}


def validate_config(config):
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact static stop-state diagnostic budget is required",
    )


def stopped_inputs(producer, checker):
    """Select only the current state and causally completed action history."""
    stream = producer["runtime"]["continuation"]
    replay = checker["runtime"]["continuation"]
    stop = stream["stop"]
    actual = stream["segments"][-1]["continuation"]["periods"]
    checked = replay["segments"][-1]["continuation"]["periods"]
    prefix.require(
        producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True
        and checker["candidate_modules_unloaded"] is True
        and stream["complete"] is False
        and stop["reason"] == "quality_no_witness"
        and stop["at_step"] == 125
        and stop["next_action_dispatched"] is False
        and replay["reported_producer_stop"] == stop
        and len(actual) == len(checked) == 5
        and [row["step"] for row in actual] == list(range(120, 125)),
        "The independently replayed five-period stopped segment is required",
    )
    last, replayed = actual[-1], checked[-1]
    state, previous = finite(last["actual_state"], (79,)).copy(), finite(last["command"], (7,)).copy()
    np.testing.assert_array_equal(state, replayed["actual_state"])
    np.testing.assert_array_equal(previous, replayed["command"])
    prefix.require(
        state_sha256(state) == last["endpoint_receipt"]["sim_state_sha256"],
        "The stopped current state must match the last completed endpoint",
    )
    plan = finite(last["decision"]["refined_plan"], (8, 6)).copy()
    np.testing.assert_array_equal(plan[0], previous[:6])
    seed = np.vstack((plan[1:], plan[-1]))
    prefix.require(
        np.max(np.abs(seed)) <= 1 + 1e-12
        and np.max(np.abs(np.diff(np.vstack((previous[:6], seed)), axis=0))) <= 0.25 + 1e-12,
        "The shifted completed plan must satisfy the original command box and rate",
    )
    past = last["completed_feedback"]["past_inner_error"]
    error = finite(past["inner_position_error_rad"], (25, 7)).copy()
    prefix.require(
        past["available"] is True
        and past["completed_periods"] == 5
        and past["total_completed_periods"] == 45
        and past["last_completed_step"] == 124
        and past["future_guarantee"] is False
        and np.all(error >= 0)
        and np.all(np.diff(error, axis=0) >= 0),
        "Only completed causal inner-error history may tighten the model",
    )
    return {"state": state, "previous": previous, "shifted_seed": seed, "past_inner_error_rad": error}


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    inputs = live.load_inputs()
    consumed = dict(inputs["source_sha256"])
    audit_ref = {"path": str(ROOT / config["cached_live_audit"]), "sha256": config["cached_live_audit_sha256"]}
    audit = prefix.read(audit_ref, consumed)
    prefix.require(
        audit["complete"] is audit["passed"] is audit["checker_policy_isolated"] is True
        and audit["control_qualified"] is audit["qualified_stable_success"] is False
        and audit["model_requests"] == 6
        and audit["denoise_calls"] == 210
        and audit["producer_stop"]["at_step"] == 125
        and audit["replayed_continuation_inner_samples"] == 1125,
        "The exact live source and independent replay audit are required",
    )
    summaries = {}
    for name, digest in (("producer", live_audit.PRODUCER_SHA), ("checker", live_audit.CHECKER_SHA)):
        ref = {"path": str(live_audit.RUN / name / "result.json"), "sha256": digest}
        summaries[name] = prefix.read(ref, consumed)
    producer = prefix.read(summaries["producer"]["branch_ref"], consumed)
    checker = prefix.read(summaries["checker"]["branch_ref"], consumed)
    stopped = stopped_inputs(producer, checker)
    runtime = producer["runtime"]
    capture = runtime["streaming_captures"][-1]
    exchange = runtime["streaming_model_exchanges"][-1]
    response = prefix.read(exchange["response_ref"], consumed)
    prefix.checked(capture["canvas_png"], consumed)
    old = inputs["old"]["old"]["base"]["previous"]
    tool = validate_transform(np.asarray(old["tool"], dtype=float))
    source = validate_transform(np.asarray(capture["current_source_pose"], dtype=float))
    prefix.require(
        capture["frame_index"] == exchange["step"] == response["step"] == 120
        and capture["canvas_png"]["sha256"] == response["image_sha256"],
        "The last reference must originate from the actual step120 image",
    )
    anchor = source @ tool
    targets, openings = decode_action_targets(anchor, response["action"], horizon=40)
    prefix.checked(config_ref, consumed)
    prefix.reference(PLAN)
    return {
        "config_ref": config_ref,
        "audit_ref": audit_ref,
        "config": config,
        "stop_inputs": stopped,
        "record_xml": old["record_xml"],
        "controller": old["controller"],
        "initial_joint": np.asarray(producer["runtime"]["runtime_initial_joint"], dtype=float),
        "tool": tool,
        "anchor": anchor,
        "targets": targets,
        "openings": openings,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "audit_ref": inputs["audit_ref"],
        "source_sha256": inputs["source_sha256"],
    }
