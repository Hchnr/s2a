"""Bind matched raw development prefixes to the frozen calibration and gripper history."""

from pathlib import Path

import numpy as np

from scripts.simulation.runtime import prepare_model_safe_tracking_fallback as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_replan_reference_switch as replan
from scripts.simulation.runtime.compare_multicase_emergency_retreat import read_array_item

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/cached_calibration_gripper_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/cached_calibration_gripper_feedback_20261111.md"
TEST = REPO / "tests/research/native_runtime/test_cached_calibration_gripper_inputs.py"


def configured_ref(ref):
    value = {"path": str(REPO / ref["path"]), "sha256": ref["sha256"]}
    prefix.checked(value)
    return value


def source_refs():
    return previous.source_refs() + [prefix.reference(path) for path in (CONFIG, PLAN, Path(__file__), TEST)]


def raw_reference(group, case):
    """The sole already-generated H40 chunk at the matched branch point."""
    return replan.selected_case(case, "new") if group == "replan" else case


def validate_chunk(case, group, chunk):
    selected = raw_reference(group, case)
    start = case["spec"]["prefix_steps"]
    prefix.require(
        chunk["step"] == start
        and chunk["model_rotation_alpha"] == 1
        and not chunk["retiming"]
        and np.asarray(chunk["effective_targets_tcp"]).shape == (40, 4, 4)
        and np.asarray(chunk["openings"]).shape == (40,)
        and np.array_equal(chunk["raw_targets_tcp"], chunk["effective_targets_tcp"]),
        "one unmodified generated H40 raw chunk required",
    )
    np.testing.assert_allclose(chunk["effective_targets_tcp"], selected["targets"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(chunk["anchor_tcp"], selected["anchor"], atol=1e-10, rtol=0)
    return selected


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        config["schema"] == "s2a.cached-calibration-gripper-feedback.v1"
        and config["case_indices"] == [2, 6, 10]
        and config["branches"] == ["raw", "calibrated"]
        and config["seed"] == 43
        and config["continuation_steps"] == 8
        and config["control_dt_s"] == 0.05
        and config["inner_dt_s"] == 0.002
        and config["joint_buffer_rad"] == 0.05
        and config["position_cap_m"] == 0.01
        and config["rotation_cap_rad"] == 0.05
        and config["arm_rate_limit"] == 0.25
        and config["braking_time_s"] == 0.1
        and config["quality_response_budget"] == 320
        and config["max_control_steps_per_phase"] == 176
        and config["max_native_steps_per_phase"] == 4400
        and config["max_new_commands_per_phase"] == 48
        and config["max_workers"] == 6
        and config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "registered matched feedback protocol differs",
    )
    candidate_ref, shadow_ref = configured_ref(config["candidate_ref"]), configured_ref(config["shadow_ref"])
    candidate, shadow = prefix.read(candidate_ref), prefix.read(shadow_ref)
    prefix.require(
        candidate["complete"] is shadow["complete"] is shadow["passed"] is True
        and shadow["physics_steps"] == shadow["new_model_calls"] == shadow["fresh_cases"] == 0,
        "frozen candidate and complete prior shadow required",
    )
    registration = prefix.read(shadow["registration_ref"])
    prefix.require(registration["inputs"]["candidate_ref"] == candidate_ref, "shadow used this exact candidate")
    existing = previous.load_inputs()
    refs = dict(existing["source_sha256"])
    prefix.checked(config_ref, refs)
    prefix.checked(candidate_ref, refs)
    prefix.checked(shadow_ref, refs)
    prefix.checked(shadow["registration_ref"], refs)
    for ref in candidate["source_refs"] + [candidate["config_ref"]]:
        prefix.checked(ref, refs)
    review = prefix.read(prefix.REVIEW, refs)
    review_map = review["source_and_artifact_sha256"]

    def reviewed(path):
        ref = prefix.reference(path)
        prefix.require(
            review_map.get(ref["path"]) == ref["sha256"], "historical gripper input was independently reviewed"
        )
        prefix.checked(ref, refs)
        return ref

    cases = {}
    for index in config["case_indices"]:
        entry = existing["cases"][index]
        group, case = entry["group"], entry["case"]
        folder = Path(review["cases"][index]["mechanical"]["case"])
        manifest_ref, controller_ref = reviewed(folder / "manifest.json"), reviewed(folder / "controller.json")
        steps_ref, chunks_ref = reviewed(folder / "steps.jsonl"), reviewed(folder / "chunks.json")
        manifest, controller = prefix.read(manifest_ref, refs), prefix.read(controller_ref, refs)
        tool_ref = reviewed(Path(manifest["runtime_config"]["tool_contract"]))
        parameters_ref = manifest["experiment"]["response_parameters_ref"]
        prefix.checked(parameters_ref, refs)
        prefix.require(
            review_map.get(parameters_ref["path"]) == parameters_ref["sha256"]
            and manifest["runtime_config"]["seed"] == 43
            and manifest["experiment"]["control_fps"] == 20
            and manifest["variant"]["governor"] is False
            and manifest["experiment"]["gripper_velocity_envelope"] == controller["gripper_velocity_envelope"]
            and manifest["experiment"]["gripper_position_hold"] == controller["gripper_config"],
            "reviewed fixed original gripper and controller settings",
        )
        chunk = read_array_item(prefix.checked(chunks_ref), case["spec"]["prefix_steps"] // 8)
        validate_chunk(case, group, chunk)
        matching = [item for item in registration["inputs"]["cases"] if item["case_id"] == case["spec"]["case_id"]]
        prefix.require(
            len(matching) == 1 and matching[0]["refs"]["chunks"] == chunks_ref, "shadow source chunk identity"
        )
        cases[index] = dict(
            group=group,
            case=case,
            gripper_refs=dict(
                manifest=manifest_ref,
                controller=controller_ref,
                steps=steps_ref,
                chunks=chunks_ref,
                tool=tool_ref,
                parameters=parameters_ref,
            ),
        )
    return dict(
        config=config,
        config_ref=config_ref,
        candidate_ref=candidate_ref,
        shadow_ref=shadow_ref,
        cases=cases,
        asset_refs=existing["asset_refs"],
        cv2_admission=existing["cv2_admission"],
        source_sha256=refs,
    )


def input_receipt(inputs):
    return dict(
        config_ref=inputs["config_ref"],
        candidate_ref=inputs["candidate_ref"],
        shadow_ref=inputs["shadow_ref"],
        cases=[
            dict(index=i, group=entry["group"], spec=entry["case"]["spec"], gripper_refs=entry["gripper_refs"])
            for i, entry in inputs["cases"].items()
        ],
        asset_refs=inputs["asset_refs"],
        source_sha256=inputs["source_sha256"],
    )
