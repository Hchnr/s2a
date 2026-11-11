"""Bind the continuous bridge study to the exact completed hold and replay."""

from pathlib import Path

from scripts.simulation.runtime import prepare_bounded_release_hold as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_bounded_release_hold_feedback as previous_runner

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/bounded_release_bridge_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/bounded_release_bridge_feedback_20261111.md"
HOLD = {
    "producer": {
        "path": str(REPO / "outputs/bounded_release_hold_feedback_20261111/producer/result.json"),
        "sha256": "04bf0a22a2a9a5867cd115965e6e475e78ec3ab64d233c4842388d54f0abe8b7",
    },
    "checker": {
        "path": str(REPO / "outputs/bounded_release_hold_feedback_20261111/checker/result.json"),
        "sha256": "3829e82d72a2734460ff383aa71faf2505b9b4b3c60ed2393c062a6f396b4a9a",
    },
}


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    expected = {
        "schema": "s2a.bounded-release-bridge-feedback.v1",
        "case_index": 5,
        "case_id": "task1_init0_s42_terminal058",
        "prefix_steps": 360,
        "old_step": 344,
        "latest_step": 360,
        "task_id": 9,
        "target": "wine_bottle_1",
        "seed": 42,
        "continuation_steps": 24,
        "hold_activation_offset": 8,
        "bridge_activation_offset": 16,
        "control_dt_s": 0.05,
        "joint_buffer_rad": 0.05,
        "position_cap_m": 0.01,
        "rotation_cap_rad": 0.05,
        "arm_rate_limit": 0.25,
        "comparison_atol": 1e-10,
        "comparison_rtol": 0,
        "max_control_steps_per_phase": 384,
        "max_native_steps_per_phase": 9600,
        "max_models_per_phase": 3,
        "max_contexts_per_phase": 3,
        "max_refiner_calls_per_phase": 24,
        "max_refiner_response_evaluations_per_phase": 11520,
        "max_quality_calls_per_phase": 24,
        "max_quality_response_evaluations_per_phase": 3840,
        "max_gripper_planner_calls_per_phase": 384,
        "max_gripper_candidate_sequences_per_phase": 2519424,
        "max_workers": 1,
        "worker_timeout_seconds": 1800,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    prefix.require(config == expected, "Exact registered continuous bridge and resource budgets")
    return ref, config


def source_refs():
    return previous_runner.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__),
            REPO / "tests/research/native_runtime/test_bounded_release_bridge_inputs.py",
        )
    ]


def load_inputs():
    config_ref, config = read_config()
    earlier = previous.load_inputs()
    case = earlier["case"]
    prefix.require(
        all(
            case["spec"][key] == config[key]
            for key in ("case_index", "case_id", "prefix_steps", "old_step", "task_id", "target")
        ),
        "The previously completed matched physical case is required",
    )
    refs = dict(earlier["source_sha256"])
    prefix.checked(config_ref, refs)
    branches = {}
    for phase in ("producer", "checker"):
        outcome = prefix.read(HOLD[phase], refs)
        prefix.require(
            outcome["complete"] is outcome["passed"] is outcome["sources_unchanged"] is True
            and outcome["phase"] == phase
            and len(outcome["branch_refs"]) == 1,
            "The earlier hold and its independent replay must be closed",
        )
        registration = prefix.read(outcome["registration_ref"], refs)
        prefix.require(registration["source_refs"] == previous_runner.source_refs(), "Earlier hold source identity")
        ref = outcome["branch_refs"][0]
        branch = prefix.read(ref, refs)
        runtime = branch["runtime"]
        prefix.require(
            branch["complete"] is branch["passed"] is True
            and branch["case"]["case_id"] == config["case_id"]
            and branch["branch"] == "bounded_release_hold"
            and runtime["continuation"]["steps_completed"] == 16
            and runtime["continuation"]["stop"] is None,
            "All earlier hold periods must have completed",
        )
        if phase == "producer":
            selection = runtime["release_selection"]
            prefix.require(
                selection["completed_hold_steps"] == 8
                and selection["bridge_executed"] is False
                and selection["state_after_execution"]["mode"] == "held"
                and selection["state_after_execution"]["staged"] is False,
                "The prior candidate reached a committed hold before its unexecuted bridge",
            )
        else:
            prefix.require(
                branch["candidate_modules_unloaded"] is True and runtime["newly_selected_commands"] == 0,
                "The prior checker independently replayed the physical commands",
            )
        branches[phase] = ref
    return {
        **earlier,
        "config_ref": config_ref,
        "config": config,
        "hold_config": earlier["config"],
        "source_sha256": refs,
        "prior_hold_refs": HOLD,
        "prior_hold_branch_refs": branches,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "case": inputs["case"]["spec"],
        "source_sha256": inputs["source_sha256"],
        "asset_refs": inputs["asset_refs"],
        "prior_hold_refs": inputs["prior_hold_refs"],
        "prior_hold_branch_refs": inputs["prior_hold_branch_refs"],
        "prior_long_refs": inputs["prior_long_refs"],
        "prior_shadow_refs": inputs["prior_shadow_refs"],
    }
