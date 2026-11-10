"""Bind a fixed component intervention to the two closed physical baselines."""

from pathlib import Path

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_bridge_feedback as bridge

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/causal_event_bridge_factorial_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/causal_event_bridge_factorial_20261110.md"
BRIDGE_RESULTS = {
    "producer": {
        "path": str(REPO / "outputs/causal_event_bridge_feedback_20261110/producer/result.json"),
        "sha256": "99f30369de8db3a7ca5d7074c21eceb73e199b26698259ef168ea3d31185a53a",
    },
    "checker": {
        "path": str(REPO / "outputs/causal_event_bridge_feedback_20261110/checker/result.json"),
        "sha256": "833d0b9331a15f98cadf2cf3e9f878c1ef6bcae4563971bf9f9878dc30465dfa",
    },
}
BRIDGE_REFERENCE = {
    "path": str(REPO / "outputs/causal_event_bridge_feedback_20261110/producer/reference.json"),
    "sha256": "4615ab0f1883c4d627288515bcf11c798053a663c86b5c7e3c9e9b4b35abe90a",
}
MODES = ("bridge_pose_old_opening", "old_pose_bridge_opening")


def source_refs():
    return bridge.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "scripts/simulation/runtime/construct_causal_reference_factorial.py",
            REPO / "scripts/verification/reference_ports/check_causal_reference_factorial.py",
            REPO / "tests/research/native_runtime/test_causal_reference_factorial.py",
            REPO / "tests/research/native_runtime/test_causal_reference_factorial_independent.py",
            REPO / "tests/research/native_runtime/test_causal_event_bridge_factorial_inputs.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == {
            "schema": "s2a.causal-event-bridge-factorial.v1",
            "case_index": 5,
            "case_id": "task1_init0_s42_terminal058",
            "prefix_steps": 360,
            "old_step": 344,
            "task_id": 9,
            "target": "wine_bottle_1",
            "seed": 42,
            "continuation_steps": 16,
            "common_steps": 8,
            "old_aligned_horizon": 24,
            "new_branches": list(MODES),
            "reused_branches": ["old_pose_old_opening", "bridge_pose_bridge_opening"],
            "control_dt_s": 0.05,
            "joint_buffer_rad": 0.05,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "comparison_atol": 1e-10,
            "comparison_rtol": 0,
            "max_control_steps_per_phase": 752,
            "max_native_steps_per_phase": 18800,
            "max_models_per_phase": 6,
            "max_contexts_per_phase": 6,
            "max_refiner_calls_per_phase": 32,
            "max_refiner_response_evaluations_per_phase": 15360,
            "max_quality_calls_per_phase": 32,
            "max_quality_response_evaluations_per_phase": 5120,
            "max_gripper_planner_calls_per_phase": 752,
            "max_gripper_candidate_sequences_per_phase": 4933872,
            "max_workers": 2,
            "worker_timeout_seconds": 1800,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered factorial cases and budgets",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    original = bridge.binding.load_inputs()
    case = original["case"]
    prefix.require(
        all(
            case["spec"][key] == config[key]
            for key in ("case_index", "case_id", "prefix_steps", "old_step", "task_id", "target")
        )
        and case["config"]["action_domain"].endswith("/libero_closed_pad_observed_endpoint_eef10_v1"),
        "Exact registered case, reference clock and LIBERO action domain",
    )
    refs = dict(original["source_sha256"])
    refs[config_ref["path"]] = config_ref["sha256"]
    prior = {}
    for phase, result_ref in BRIDGE_RESULTS.items():
        result = prefix.read(result_ref, refs)
        prefix.require(
            result["complete"] is result["passed"] is result["sources_unchanged"] is True
            and result["phase"] == phase
            and result["reference_ref"] == BRIDGE_REFERENCE
            and result["world_model_calls"] == result["fresh_cases"] == 0,
            "Original bridge producer and independent checker closed",
        )
        item = prefix.read(result["branch_ref"], refs)
        continuation = item["runtime"]["continuation"]
        prefix.require(
            item["complete"] is item["passed"] is True
            and item["phase"] == phase
            and item["case"]["case_id"] == config["case_id"]
            and item["runtime"]["previous_verified_steps"] == config["common_steps"]
            and continuation["steps_completed"] == config["continuation_steps"]
            and continuation["stop"] is None,
            "Closed bridge branch executed all sixteen actions",
        )
        prior[phase] = result["branch_ref"]
    reference_document = prefix.read(BRIDGE_REFERENCE, refs)
    bridge_reference, audit = bridge.checked_reference(original, reference_document)
    old, latest = bridge.binding.selected_references(case)
    prefix.require(
        old["cursor"] == config["prefix_steps"] - config["old_step"]
        and latest["cursor"] == 0
        and audit["passed"] is True,
        "Both generated inputs and independent bridge geometry verified",
    )
    old_ref = original["prior_branches"]["remembered"]["producer_ref"]
    old_checker_ref = original["prior_branches"]["remembered"]["checker_ref"]
    for phase, ref in (("producer", old_ref), ("checker", old_checker_ref)):
        old_result = prefix.read(ref, refs)
        prefix.require(
            old_result["complete"] is old_result["passed"] is True
            and old_result["phase"] == phase
            and old_result["runtime"]["continuation"]["steps_completed"] == 16
            and old_result["runtime"]["continuation"]["stop"] is None,
            "Closed original old-reference baseline",
        )
    return {
        "config_ref": config_ref,
        "config": config,
        "case": case,
        "source_sha256": refs,
        "asset_refs": original["asset_refs"],
        "cv2_admission": original["cv2_admission"],
        "bridge_reference": bridge_reference,
        "bridge_reference_ref": BRIDGE_REFERENCE,
        "bridge_baselines": prior,
        "old_baselines": {"producer": old_ref, "checker": old_checker_ref},
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "case": inputs["case"]["spec"],
        "source_sha256": inputs["source_sha256"],
        "asset_refs": inputs["asset_refs"],
        "bridge_reference_ref": inputs["bridge_reference_ref"],
        "bridge_baselines": inputs["bridge_baselines"],
        "old_baselines": inputs["old_baselines"],
    }


def component_arguments(inputs, mode):
    prefix.require(mode in MODES, "One registered mixed-component branch")
    old, _latest = bridge.binding.selected_references(inputs["case"])
    reference, config = inputs["bridge_reference"], inputs["config"]
    return (
        old["targets"],
        old["openings"],
        reference["poses"],
        reference["openings"],
    ), {
        "old_step": config["old_step"],
        "origin_step": config["prefix_steps"],
        "common_steps": config["common_steps"],
        "continuation_steps": config["continuation_steps"],
        "preview_steps": 8,
        "mode": mode,
    }


def construct_reference(inputs, mode):
    """Only a producer imports the component intervention constructor."""
    from scripts.simulation.runtime.construct_causal_reference_factorial import construct_reference as construct

    args, kwargs = component_arguments(inputs, mode)
    return construct(*args, **kwargs)


def verify_reference(inputs, mode, reference):
    from scripts.verification.reference_ports.check_causal_reference_factorial import verify_reference as verify

    args, kwargs = component_arguments(inputs, mode)
    audit = verify(*args, reference, **kwargs)
    prefix.require(audit["old_aligned_horizon"] == inputs["config"]["old_aligned_horizon"], "Fixed old-chunk coverage")
    return audit
