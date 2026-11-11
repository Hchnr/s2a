"""Admit one source-pinned, same-reference physical face-projection probe."""

import numpy as np

from scripts.simulation.runtime import prepare_early_tabletop_live_feedback as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_early_tabletop_live_feedback as audit

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/box_face_feedback_20261111.json"
CONFIG_SHA = "b2f9a2a74edb03c65655f3e0e20f5a1edac48be22666f089991044fa066e6c4a"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/box_face_feedback_20261111.md"
OUTPUT = ROOT / "outputs/box_face_feedback_20261111"
DRY = ROOT / "outputs/box_face_feedback_dry_20261111/result.json"
QA = ROOT / "outputs/qa/box_face_feedback_20261111/result.json"


def validate_config(config):
    prefix.require(
        config["schema"] == "s2a.box-face-feedback-study.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and (config["case_index"], config["seed"], config["mode"], config["model_step"]) == (8, 42, "original", 120)
        and config["new_period_limit"] == 6
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
        == (126, 3150, 6300)
        and (config["max_new_commands"], config["max_native_models"], config["max_software_contexts"]) == (6, 6, 6)
        and config["new_model_initializations"] == config["new_model_requests"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The exact original reference, policy conditions and physical budgets are required",
    )


def load_inputs():
    inputs = previous.load_inputs()
    consumed = dict(inputs["early_tabletop_live_feedback"]["source_sha256"])
    config_ref = {"path": str(CONFIG), "sha256": CONFIG_SHA}
    config = prefix.read(config_ref, consumed)
    validate_config(config)
    names = ("static_result", "baseline_owner", "baseline_producer", "baseline_checker")
    refs = {name: {"path": str(ROOT / config[name]), "sha256": config[name + "_sha256"]} for name in names}
    static, owner, producer, checker = (prefix.read(refs[name], consumed) for name in names)
    prefix.require(
        static["complete"] is static["passed"] is static["sources_unchanged"] is True
        and static["native_models"] == static["forward_queries"] == static["physics_steps"] == 0
        and static["numerical_endpoints"] == 16
        and owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True
        and owner["producer_refs"]["original"] == refs["baseline_producer"]
        and owner["checker_refs"]["original"] == refs["baseline_checker"]
        and producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True,
        "The source-pinned static study and independently replayed original branch are required",
    )
    for ref in static["source_refs"]:
        prefix.checked(ref, consumed)
    audit.check_pair(producer, checker)
    old = inputs["early_tabletop_live_feedback"]["branches"]["original"]
    continuation = producer["runtime"]["continuation"]
    prefix.require(
        old["state_sha256"] == continuation["model_state_sha256"] == config["model_state_sha256"]
        and old["image_ref"]["sha256"] == continuation["model_image_ref"]["sha256"] == config["model_image_sha256"]
        and old["response_ref"]["sha256"]
        == continuation["model_response_ref"]["sha256"]
        == config["model_response_sha256"]
        and [row["step"] for row in continuation["probe"]["periods"]] == list(range(120, 128))
        and len(continuation["inner_samples"]) == 200,
        "The historical physical reference must match the original branch's state and image",
    )
    rows = {(row["mode"], row["step"]): row for row in static["analysis"]["rows"]}
    prefix.require(
        len(rows) == 16 and rows[("original", 120)]["state_sha256"] == old["state_sha256"],
        "The static analysis must use the same original start",
    )
    boxes = static["analysis"]["box_anchors"]
    prefix.require([entry[0] for entry in boxes] == ["table", "stove"], "Previously verified box identities required")
    for entry, name in zip(boxes, ("table", "stove"), strict=True):
        np.testing.assert_allclose(entry[1], config[name + "_pose"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(entry[2], config[name + "_half_size_m"], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(static["analysis"]["joint_interval_m"], config["finger_joint_range_m"])
    inputs["box_face_feedback"] = {
        "config": config,
        "config_ref": config_ref,
        "refs": refs,
        "baseline_producer": producer,
        "baseline_checker": checker,
        "source_sha256": consumed,
    }
    return inputs


def input_receipt(inputs):
    item = inputs["box_face_feedback"]
    return {
        "prior": previous.input_receipt(inputs),
        "config_ref": item["config_ref"],
        "refs": item["refs"],
        "source_sha256": item["source_sha256"],
    }
