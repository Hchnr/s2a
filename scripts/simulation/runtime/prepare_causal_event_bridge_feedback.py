"""Bind the registered transition to independently replayed causal inputs."""

from pathlib import Path

from scripts.simulation.runtime import prepare_causal_event_feedback as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_switch_feedback as switch

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/causal_event_bridge_feedback_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/causal_event_bridge_feedback_20261110.md"
SWITCH_RESULTS = {
    "producer": {
        "path": str(REPO / "outputs/causal_event_switch_feedback_20261110/producer/result.json"),
        "sha256": "8a0a2f95d71df025ffd10901b263e03db1e7d48824554cc92c26f9f6c38ec023",
    },
    "checker": {
        "path": str(REPO / "outputs/causal_event_switch_feedback_20261110/checker/result.json"),
        "sha256": "86406034f61f04864475a584ad5959bad87d85401c986cd03ad2ffe32e6daf4f",
    },
}


def source_refs():
    return switch.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "src/s2a/research/wam/tracking/preview/causal_reference_bridge.py",
            REPO / "scripts/verification/reference_ports/check_causal_reference_bridge.py",
            REPO / "tests/research/wam/tracking/preview/test_causal_reference_bridge.py",
            REPO / "tests/research/wam/tracking/preview/test_causal_reference_bridge_independent.py",
            REPO / "tests/research/native_runtime/test_causal_event_bridge_inputs.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == {
            "schema": "s2a.causal-event-bridge-feedback.v1",
            "case_index": 5,
            "case_id": "task1_init0_s42_terminal058",
            "prefix_steps": 360,
            "old_step": 344,
            "task_id": 9,
            "target": "wine_bottle_1",
            "seed": 42,
            "continuation_steps": 16,
            "bridge_start_offset": 8,
            "bridge_steps": 8,
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
        },
        "Exact registered causal transition and budget",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    inputs = switch.load_inputs()
    case = inputs["case"]
    prefix.require(
        all(
            case["spec"][key] == config[key]
            for key in ("case_index", "case_id", "prefix_steps", "old_step", "task_id", "target")
        ),
        "Registered case and generated reference times",
    )
    refs = dict(inputs["source_sha256"])
    refs[config_ref["path"]] = config_ref["sha256"]
    direct_branches = {}
    for phase, result_ref in SWITCH_RESULTS.items():
        result = prefix.read(result_ref, refs)
        prefix.require(
            result["complete"] is result["passed"] is result["sources_unchanged"] is True
            and result["phase"] == phase
            and result["world_model_calls"] == result["fresh_cases"] == 0,
            "Earlier direct-switch phase closed without a new world model call",
        )
        branch_ref = result["branch_ref"]
        branch = prefix.read(branch_ref, refs)
        runtime = branch["runtime"]
        continuation = runtime["continuation"]
        stop = continuation["stop"]
        prefix.require(
            branch["complete"] is branch["passed"] is True
            and branch["phase"] == phase
            and branch["case"]["case_id"] == config["case_id"]
            and runtime["previous_verified_steps"] == continuation["steps_completed"] == 8
            and stop["reason"] == "quality_no_witness"
            and stop["at_step"] == 368
            and stop["next_action_dispatched"] is False,
            "Earlier direct switch stopped before the ninth action",
        )
        if phase == "producer":
            prefix.require(runtime["reference_switch"]["completed"] is True, "Earlier reference switch completed")
        else:
            prefix.require(
                continuation["switch_decision_not_recomputed"]
                is continuation["decision_stop_recorded_not_recomputed"]
                is True,
                "Earlier checker explicitly did not recompute the decision",
            )
        direct_branches[phase] = branch_ref
    return {
        "config_ref": config_ref,
        "config": config,
        "case": case,
        "source_sha256": refs,
        "asset_refs": inputs["asset_refs"],
        "cv2_admission": inputs["cv2_admission"],
        "prior_branches": inputs["prior_branches"],
        "direct_branches": direct_branches,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "case": inputs["case"]["spec"],
        "source_sha256": inputs["source_sha256"],
        "asset_refs": inputs["asset_refs"],
        "prior_branches": inputs["prior_branches"],
        "direct_branches": inputs["direct_branches"],
    }


def selected_references(case):
    old, latest = (binding.selected_case(case, mode, horizon=16) for mode in ("remembered", "latest"))
    prefix.require(old["cursor"] == 16 and latest["cursor"] == 0, "Causally aligned saved reference chunks")
    return old, latest


def prepare_reference(case, config):
    """Only the producer imports and constructs the transition."""
    from s2a.research.wam.tracking.preview.causal_reference_bridge import build_reference_bridge

    old, latest = selected_references(case)
    return build_reference_bridge(
        old["targets"],
        old["openings"],
        latest["targets"],
        latest["openings"],
        old_step=config["old_step"],
        latest_step=config["prefix_steps"],
        start_offset=config["bridge_start_offset"],
        transition_steps=config["bridge_steps"],
    )


def verify_reference(case, config, reference):
    """Independently check the recorded transition against the source chunks."""
    from scripts.verification.reference_ports.check_causal_reference_bridge import verify_reference_bridge

    old, latest = selected_references(case)
    return verify_reference_bridge(
        old["targets"],
        old["openings"],
        latest["targets"],
        latest["openings"],
        reference,
        old_step=config["old_step"],
        latest_step=config["prefix_steps"],
        start_offset=config["bridge_start_offset"],
        transition_steps=config["bridge_steps"],
        atol=config["comparison_atol"],
    )
