"""Bind a bounded release-hold pilot to earlier physical and shadow evidence."""

from pathlib import Path

from scripts.simulation.runtime import prepare_causal_event_bridge_feedback as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_bridge_feedback as previous_runner
from scripts.wam.tracking.preview import bind_bounded_release_shadow as shadow

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/bounded_release_hold_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/bounded_release_hold_feedback_20261111.md"
SHADOW = {
    "producer": {
        "path": str(REPO / "outputs/bounded_release_revision_shadow_20261111/producer/result.json"),
        "sha256": "cbde160b26f3590bfd0e14614acd9dfa5aa3ca698ac4f3b3502a897d8d0f3881",
    },
    "checker": {
        "path": str(REPO / "outputs/bounded_release_revision_shadow_20261111/checker/result.json"),
        "sha256": "a233813716675decc497f7d3641d85f8f8f5fb337779d1f8e5cc7e5463992d5d",
    },
}
LONG = {
    "producer": {
        "path": str(REPO / "outputs/causal_event_long_feedback_20261110/producer/remembered/result.json"),
        "sha256": "ca7b9be38fde1568dd578625bf7546635ad67400ced38781ab157ccde256e3f3",
    },
    "checker": {
        "path": str(REPO / "outputs/causal_event_long_feedback_20261110/checker/remembered/result.json"),
        "sha256": "a0e7d092ac615f9ee2fa3764c6d23ff314067ea6a8a16cf432fbecf05d1b0be9",
    },
}
BRIDGE_REGISTRATION = prefix.REPO / "outputs/causal_event_bridge_feedback_20261110/producer/registration.json"


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    expected = {
        "schema": "s2a.bounded-release-hold-feedback.v1",
        "case_index": 5,
        "case_id": "task1_init0_s42_terminal058",
        "prefix_steps": 360,
        "old_step": 344,
        "latest_step": 360,
        "task_id": 9,
        "target": "wine_bottle_1",
        "seed": 42,
        "continuation_steps": 16,
        "activation_offset": 8,
        "control_dt_s": 0.05,
        "joint_buffer_rad": 0.05,
        "position_cap_m": 0.01,
        "rotation_cap_rad": 0.05,
        "arm_rate_limit": 0.25,
        "comparison_atol": 1e-10,
        "comparison_rtol": 0,
        "max_control_steps_per_phase": 376,
        "max_native_steps_per_phase": 9400,
        "max_models_per_phase": 3,
        "max_contexts_per_phase": 3,
        "max_refiner_calls_per_phase": 16,
        "max_refiner_response_evaluations_per_phase": 7680,
        "max_quality_calls_per_phase": 16,
        "max_quality_response_evaluations_per_phase": 2560,
        "max_gripper_planner_calls_per_phase": 376,
        "max_gripper_candidate_sequences_per_phase": 2466936,
        "max_workers": 1,
        "worker_timeout_seconds": 1800,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    prefix.require(config == expected, "Exact registered causal hold pilot and budgets")
    return ref, config


def source_refs():
    return (
        previous_runner.source_refs()
        + shadow.source_refs()
        + [
            prefix.reference(path)
            for path in (
                CONFIG,
                PLAN,
                Path(__file__),
                REPO / "tests/research/native_runtime/test_bounded_release_hold_inputs.py",
            )
        ]
    )


def load_inputs():
    config_ref, config = read_config()
    earlier = previous.load_inputs()
    case = earlier["case"]
    prefix.require(
        all(
            case["spec"][key] == config[key]
            for key in ("case_index", "case_id", "prefix_steps", "old_step", "task_id", "target")
        ),
        "Original matched case, generated references and prefix",
    )
    old_registration_ref = prefix.reference(BRIDGE_REGISTRATION)
    old_registration = prefix.read(old_registration_ref)
    prefix.require(old_registration["source_refs"] == previous_runner.source_refs(), "Earlier bridge source identity")
    prior, prior_shadow = {}, {}
    for phase in ("producer", "checker"):
        branch = prefix.read(LONG[phase])
        prefix.require(
            branch["complete"] is branch["passed"] is True
            and branch["case"]["case_id"] == config["case_id"]
            and branch["branch"] == "remembered"
            and branch["runtime"]["continuation"]["steps_completed"] == 16
            and branch["runtime"]["continuation"]["stop"] is None,
            "Both earlier 16-period old-reference physical branches closed",
        )
        prior[phase] = LONG[phase]
        outcome = prefix.read(SHADOW[phase])
        reg = prefix.read(outcome["registration_ref"])
        prefix.require(
            outcome["complete"] is outcome["passed"] is outcome["sources_unchanged"] is True
            and outcome["phase"] == phase
            and outcome["studied_summary"]["step"] == 368
            and outcome["studied_summary"]["reason"] == "bounded_hold"
            and reg["source_refs"] == shadow.source_refs(),
            "Prior shadow and independent check with unchanged candidate source",
        )
        prior_shadow[phase] = SHADOW[phase]
    old, latest = previous.selected_references(case)
    prefix.require(old["cursor"] == 16 and latest["cursor"] == 0, "Original reference time alignment")
    refs = dict(earlier["source_sha256"])
    for ref in (config_ref, old_registration_ref, *prior.values(), *prior_shadow.values()):
        prefix.checked(ref, refs)
    return {
        "config_ref": config_ref,
        "config": config,
        "case": case,
        "source_sha256": refs,
        "asset_refs": earlier["asset_refs"],
        "cv2_admission": earlier["cv2_admission"],
        "prior_long_refs": prior,
        "prior_shadow_refs": prior_shadow,
        "old_registration_ref": old_registration_ref,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "case": inputs["case"]["spec"],
        "source_sha256": inputs["source_sha256"],
        "asset_refs": inputs["asset_refs"],
        "prior_long_refs": inputs["prior_long_refs"],
        "prior_shadow_refs": inputs["prior_shadow_refs"],
        "old_registration_ref": inputs["old_registration_ref"],
    }
