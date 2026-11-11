"""Admit an earlier two-branch probe using the same saved step112 reference."""

import numpy as np

from scripts.simulation.runtime import prepare_early_tabletop_intervention as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_early_tabletop_capture as capture
from scripts.simulation.runtime import run_tabletop_projection_feedback_probe as audit

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/early_box_face_feedback_20261111.json"
CONFIG_SHA = "fae2224883494eb3428e2586429a43116e3caec35f50d62d8241d34d4fd41e28"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/early_box_face_feedback_20261111.md"
OUTPUT = ROOT / "outputs/early_box_face_feedback_20261111"
DRY = ROOT / "outputs/early_box_face_feedback_dry_20261111/result.json"
QA = ROOT / "outputs/qa/early_box_face_feedback_20261111/result.json"


def validate_config(config):
    prefix.require(
        config["schema"] == "s2a.early-box-face-feedback-study.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and (config["case_index"], config["seed"], config["branch_step"], config["reference_step"]) == (8, 42, 118, 112)
        and config["modes"] == ["original", "box_face"]
        and config["new_period_limit"] == 8
        and (config["table_geom"], config["stove_geom"], config["finger_geom"], config["finger_joint"])
        == ("table_collision", "flat_stove_1_collision_burner", "gripper0_finger2_collision", "gripper0_finger_joint2")
        and (
            config["position_uncertainty_m"],
            config["rotation_uncertainty_rad"],
            config["additional_shape_uncertainty_m"],
        )
        == (0.01, 0.05, 0.0)
        and (config["clearance_m"], config["numerical_margin_m"], config["max_translation_m"]) == (0.001, 1e-9, 0.05)
        and (config["command_step_limit"], config["joint_buffer_rad"], config["controller_dt_s"], config["native_dt_s"])
        == (0.25, 0.05, 0.05, 0.002)
        and (
            config["max_control_steps_per_phase"],
            config["max_native_steps_per_phase"],
            config["max_total_native_steps"],
        )
        == (126, 3150, 12600)
        and (config["max_new_commands"], config["max_native_models"], config["max_software_contexts"]) == (16, 12, 12)
        and config["new_model_initializations"] == config["new_model_requests"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The exact earlier common reference, two modes and physical budgets are required",
    )


def load_inputs():
    inputs = previous.load_inputs()
    old = inputs["early_tabletop_intervention"]
    consumed = dict(old["source_sha256"])
    config_ref = {"path": str(CONFIG), "sha256": CONFIG_SHA}
    config = prefix.read(config_ref, consumed)
    validate_config(config)
    names = ("capture_owner", "baseline_producer", "baseline_checker")
    refs = {name: {"path": str(ROOT / config[name]), "sha256": config[name + "_sha256"]} for name in names}
    owner, producer, checker = (prefix.read(refs[name], consumed) for name in names)
    prefix.require(
        owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True
        and owner["producer_refs"]["original"] == refs["baseline_producer"]
        and owner["checker_refs"]["original"] == refs["baseline_checker"]
        and producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True,
        "The independently replayed early branch and original source must be closed",
    )
    audit.check_pair(producer, checker)
    capture.compare_capture(producer, checker)
    start = producer["runtime"]["continuation"]["branch_start"]
    prefix.require(
        start["state_sha256"] == config["branch_state_sha256"] == old["config"]["branch_state_sha256"]
        and old["source_capture_ref"]["sha256"] == config["reference_image_sha256"]
        and old["source_response_ref"]["sha256"] == config["reference_response_sha256"]
        and old["config"]["source_reference_step"] == config["reference_step"]
        and len(old["source_reference"]) == 40,
        "The exact step118 state and step112 current-image WAM reference are required",
    )
    np.testing.assert_allclose(old["config"]["table_pose"], config["table_pose"], atol=0, rtol=0)
    np.testing.assert_allclose(old["config"]["table_half_size_m"], config["table_half_size_m"], atol=0, rtol=0)
    inputs["early_box_face_feedback"] = {
        "config": config,
        "config_ref": config_ref,
        "refs": refs,
        "baseline_start": start,
        "source_sha256": consumed,
    }
    return inputs


def input_receipt(inputs):
    item = inputs["early_box_face_feedback"]
    return {
        "prior": previous.input_receipt(inputs),
        "config_ref": item["config_ref"],
        "refs": item["refs"],
        "source_sha256": item["source_sha256"],
    }
