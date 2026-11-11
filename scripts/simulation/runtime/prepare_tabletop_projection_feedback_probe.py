"""Pin recorded state, response, controls and budgets for the tabletop probe."""

import copy
from pathlib import Path

import numpy as np

from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import causal_stream_live_48_replay as references
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as recorded
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_wam_two_window_feedback as historical
from scripts.tracking.response_models import reference_table_geometry as geometry

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/tabletop_projection_feedback_probe_20261111.json"
CONFIG_SHA = "c4de89e742a0a0502058b4d3d79517d849d394a14918a1d221aaf3346293f714"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/tabletop_projection_feedback_probe_20261111.md"
OUTPUT = ROOT / "outputs/tabletop_projection_feedback_probe_20261111"
DRY = ROOT / "outputs/tabletop_projection_feedback_probe_dry_20261111/result.json"
QA = ROOT / "outputs/qa/tabletop_projection_feedback_probe_20261111/result.json"


def replay_prefix_stream(stream):
    """Derive five complete pre-branch segments without altering source evidence."""
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
        "Forty consecutive fully completed pre-branch periods are required",
    )
    return {
        "segments": segments,
        "complete": True,
        "stop": None,
        "control_steps_attempted": 40,
        "control_steps_completed": 40,
        "feedback_periods_committed": 40,
    }


def load_inputs():
    inputs = geometry.load_inputs()
    consumed = dict(inputs["reference_table_geometry"]["source_sha256"])
    config = prefix.read({"path": str(CONFIG), "sha256": CONFIG_SHA}, consumed)
    prefix.require(
        config["schema"] == "s2a.tabletop-projection-feedback-probe.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and config["case_index"] == 8
        and config["seed"] == 42
        and (
            config["prefix_steps"],
            config["replayed_periods"],
            config["branch_step"],
            config["new_periods_per_producer"],
        )
        == (80, 40, 120, 7)
        and config["branch_state_sha256"] == "977c527f571f3b03b56f66118612027a8d4835c5fc9f9b9afea637f4a029e33c"
        and config["modes"] == ["original", "nominal", "pose_bounded"]
        and config["table_geom"] == "table_collision"
        and config["finger_geom"] == "gripper0_finger2_collision"
        and config["table_pose"] == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.875], [0, 0, 0, 1]]
        and config["table_half_size_m"] == [0.5, 0.6, 0.025]
        and (config["clearance_m"], config["numerical_margin_m"], config["max_translation_m"]) == (0.001, 1e-9, 0.05)
        and (config["command_step_limit"], config["joint_buffer_rad"], config["controller_dt_s"], config["native_dt_s"])
        == (0.25, 0.05, 0.05, 0.002)
        and (config["max_control_steps_per_phase"], config["max_native_steps_per_phase"], config["max_new_commands"])
        == (127, 3175, 21)
        and config["max_native_models"] == config["max_software_contexts"] == 18
        and config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The exact pre-registered branch, observations and physical budget are required",
    )
    names = ("previous_producer", "previous_checker", "previous_audit", "static_audit")
    refs = {key: {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]} for key in names}
    producer, checker, prior_audit, static_audit = (prefix.read(refs[key], consumed) for key in names)
    prefix.require(
        all(result["complete"] is result["passed"] is True for result in (producer, checker, prior_audit, static_audit))
        and prior_audit["checker_policy_isolated"] is True
        and prior_audit["control_qualified"] is static_audit["control_qualified"] is False,
        "The original independently replayed stream and static audit are required",
    )
    original_rows, replay_rows = recorded.periods(producer), recorded.periods(checker)
    prefix.require(len(original_rows) == len(replay_rows) == 46, "The complete earlier physical replay is required")
    for step, (original, replay) in enumerate(zip(original_rows[:40], replay_rows[:40], strict=True), start=80):
        prefix.require(original["step"] == replay["step"] == step, "Consecutive pre-branch periods required")
        historical.compare_first_period(original, replay)
    replay = replay_prefix_stream(producer["runtime"]["continuation"])
    source_root = Path(refs["previous_producer"]["path"]).parents[2] / "producer/simulation/runtime"
    chunks, observations = references.decode_actual_references(inputs, producer, image_root=source_root, max_windows=6)
    prefix.require(len(chunks) == len(observations) == 6, "Six source-verified generated references required")
    for offset, target in enumerate(inputs["reference_table_geometry"]["targets"]):
        np.testing.assert_allclose(target, chunks[offset // 8][offset % 8], atol=1e-12, rtol=0)
    capture = producer["runtime"]["streaming_captures"][5]
    prefix.require(
        capture["frame_index"] == 120
        and capture["current_endpoint"]["sim_state_sha256"] == config["branch_state_sha256"]
        and state_sha256(inputs["reference_table_geometry"]["states"][40]) == config["branch_state_sha256"]
        and original_rows[39]["endpoint_receipt"]["sim_state_sha256"] == config["branch_state_sha256"],
        "The new reference must belong to the exact common branch state",
    )
    inputs["tabletop_projection_feedback_probe"] = {
        "config": config,
        "refs": refs,
        "source_sha256": consumed,
        "previous_producer": producer,
        "previous_checker": checker,
        "prefix_stream": replay,
        "target_chunks": chunks,
        "source_observation": observations[5],
        "previous_command": np.asarray(original_rows[39]["command"], dtype=float).copy(),
    }
    return inputs


def input_receipt(inputs):
    item = inputs["tabletop_projection_feedback_probe"]
    return {
        "prior": geometry.input_receipt(inputs),
        "config_ref": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "refs": item["refs"],
        "source_sha256": item["source_sha256"],
        "branch_state_sha256": item["config"]["branch_state_sha256"],
    }
