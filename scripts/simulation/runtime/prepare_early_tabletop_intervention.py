"""Bind the pre-contact branch to an already generated, audited H40."""

import copy

import numpy as np

from s2a.references.action_reference import decode_action_targets
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as recorded
from scripts.simulation.runtime import prepare_tabletop_projection_feedback_probe as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/early_tabletop_intervention_live_wam_20261111.json"
CONFIG_SHA = "82a67afe50153acc2badc4a9b0407603a6b470ca398665657f5bc959cbb7f754"
PLAN = ROOT / "plans/v0.0.1/tracking/references/early_tabletop_intervention_live_wam_20261111.md"
OUTPUT = ROOT / "outputs/early_tabletop_intervention_live_wam_20261111"


def replay_prefix_stream(stream):
    """Derive the completed 38-period prefix and stop before step118."""
    prefix.require(type(stream) is dict and len(stream.get("segments", [])) == 6, "Six source segments required")
    segments = copy.deepcopy(stream["segments"][:5])
    prefix.require(
        [row["start_step"] for row in segments] == [80, 88, 96, 104, 112]
        and all(
            row["inflight"] is None
            and row["continuation"]["steps_completed"] == 8
            and len(row["continuation"]["periods"]) == 8
            and row["continuation"]["stop"] is None
            for row in segments
        ),
        "Forty complete original periods are required before deriving the branch prefix",
    )
    last = segments[-1]["continuation"]
    last["periods"] = last["periods"][:6]
    last["steps_completed"] = 6
    last["stop"] = {"reason": "registered_branch_boundary", "next_action_dispatched": False}
    return {
        "segments": segments,
        "complete": False,
        "stop": {"reason": "registered_branch_boundary", "at_step": 118, "next_action_dispatched": False},
        "control_steps_attempted": 38,
        "control_steps_completed": 38,
        "feedback_periods_committed": 38,
    }


def load_inputs():
    inputs = previous.load_inputs()
    prior = inputs["tabletop_projection_feedback_probe"]
    consumed = dict(prior["source_sha256"])
    config = prefix.read({"path": str(CONFIG), "sha256": CONFIG_SHA}, consumed)
    prefix.require(
        config["schema"] == "s2a.early-tabletop-intervention-live-wam.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and (config["case_index"], config["seed"], config["prefix_steps"], config["replayed_periods"])
        == (8, 42, 80, 38)
        and (config["branch_step"], config["source_reference_step"], config["fresh_reference_step"]) == (118, 112, 120)
        and config["branch_state_sha256"] == "c82cabe680cbf205c0a9555c393b8f53eb96bbb97ca74c8b5c9bd3f675da8b82"
        and (config["early_periods"], config["live_periods"]) == (2, 8)
        and config["modes"] == ["original", "pose_bounded"]
        and (config["clearance_m"], config["numerical_margin_m"], config["max_translation_m"]) == (0.001, 1e-9, 0.05)
        and (config["command_step_limit"], config["joint_buffer_rad"], config["controller_dt_s"], config["native_dt_s"])
        == (0.25, 0.05, 0.05, 0.002)
        and config["table_pose"] == prior["config"]["table_pose"]
        and config["table_half_size_m"] == prior["config"]["table_half_size_m"]
        and (config["table_geom"], config["finger_geom"]) == ("table_collision", "gripper0_finger2_collision")
        and (config["stage_a_max_control_steps_per_phase"], config["stage_a_max_native_steps_per_phase"]) == (120, 3000)
        and (config["stage_b_max_control_steps_per_phase"], config["stage_b_max_native_steps_per_phase"]) == (128, 3200)
        and config["max_native_models"] == config["max_software_contexts"] == 24
        and config["max_new_commands"] == 20
        and config["gpu_uuid"] == "GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf"
        and (config["model_initializations"], config["model_requests"], config["denoise_calls_per_request"])
        == (1, 2, 35)
        and config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The fixed causal branch, model and physical budgets are required",
    )
    names = ("previous_physical", "previous_summary_audit")
    refs = {key: {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]} for key in names}
    physical, summary = (prefix.read(refs[key], consumed) for key in names)
    prefix.require(
        physical["complete"] is physical["passed"] is summary["complete"] is summary["passed"] is True
        and physical["control_qualified"] is summary["control_qualified"] is False
        and summary["owner_ref"] == refs["previous_physical"],
        "The independently replayed physical comparison and summary correction are required",
    )
    producer = prior["previous_producer"]
    rows = recorded.periods(producer)
    prefix.require(
        rows[37]["step"] == 117
        and rows[37]["endpoint_receipt"]["sim_state_sha256"] == config["branch_state_sha256"]
        and state_sha256(inputs["reference_table_geometry"]["states"][38]) == config["branch_state_sha256"],
        "The branch must match the original completed step117 state",
    )
    capture = producer["runtime"]["streaming_captures"][4]
    exchange = producer["runtime"]["streaming_model_exchanges"][4]
    response = prefix.read(exchange["response_ref"], consumed)
    prefix.checked(capture["canvas_png"], consumed)
    prefix.require(
        capture["frame_index"] == exchange["step"] == response["step"] == config["source_reference_step"]
        and response["image_sha256"] == capture["canvas_png"]["sha256"]
        and response["action_dim_mask"] == [True] * 10 + [False] * 118,
        "The H40 must come from the audited step112 current image",
    )
    tool = np.asarray(inputs["old"]["old"]["base"]["previous"]["tool"])
    poses, openings = decode_action_targets(
        np.asarray(capture["current_source_pose"]) @ tool, np.asarray(response["action"]), horizon=40
    )
    np.testing.assert_allclose(poses[:8], prior["target_chunks"][4], atol=1e-12, rtol=0)
    inputs["early_tabletop_intervention"] = {
        "config": config,
        "refs": refs,
        "source_sha256": consumed,
        "prefix_stream": replay_prefix_stream(producer["runtime"]["continuation"]),
        "initial_command": np.asarray(rows[37]["command"], dtype=float).copy(),
        "source_reference": poses,
        "source_openings": openings,
        "source_capture_ref": capture["canvas_png"],
        "source_response_ref": exchange["response_ref"],
    }
    return inputs


def input_receipt(inputs):
    item = inputs["early_tabletop_intervention"]
    return {
        "prior": previous.input_receipt(inputs),
        "config_ref": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "refs": item["refs"],
        "source_capture_ref": item["source_capture_ref"],
        "source_response_ref": item["source_response_ref"],
        "source_sha256": item["source_sha256"],
    }
