"""Bind one previously studied release window and its independent baseline."""

from pathlib import Path

import numpy as np

from scripts.simulation.runtime import prepare_causal_opening_feedback as earlier
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/gripper/defer_close_task1_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/gripper/defer_close_task1_feedback_20261111.md"


def source_refs():
    return earlier.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__),
            REPO / "tests/research/native_runtime/test_defer_close_task1_inputs.py",
        )
    ]


def read_config():
    prefix.environment()
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    fixed = dict(
        schema="s2a.defer-close-task1-feedback.v1",
        output="outputs/defer_close_task1_feedback_20261111",
        case_index=7,
        task_id=9,
        seed=43,
        prefix_steps=232,
        continuation_steps=8,
        target_object="wine_bottle_1",
        control_dt_s=0.05,
        inner_dt_s=0.002,
        joint_buffer_rad=0.05,
        position_cap_m=0.01,
        rotation_cap_rad=0.05,
        arm_rate_limit=0.25,
        max_control_steps_per_phase=240,
        max_native_steps_per_phase=6000,
        max_gripper_planner_calls=240,
        max_gripper_candidate_sequences=1574640,
        max_refiner_calls=8,
        max_refiner_responses=3840,
        max_quality_calls=8,
        max_quality_responses=1280,
        max_workers=1,
        worker_timeout_seconds=1800,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    prefix.require(
        all(config.get(key) == value and type(config.get(key)) is type(value) for key, value in fixed.items()),
        "Fixed task1 feedback protocol required",
    )
    return ref, config


def configured_ref(config, name):
    ref = {"path": str(REPO / config[name]), "sha256": config[name + "_sha256"]}
    prefix.checked(ref)
    return ref


def load_inputs():
    config_ref, config = read_config()
    names = (
        "baseline_producer",
        "baseline_checker",
        "baseline_branch",
        "baseline_branch_checker",
        "shadow",
        "correct_producer",
        "correct_checker",
    )
    refs = {name: configured_ref(config, name) for name in names}
    documents = {name: prefix.read(ref) for name, ref in refs.items()}
    for name in ("baseline_producer", "baseline_checker", "correct_producer", "correct_checker"):
        document = documents[name]
        prefix.require(
            document["complete"] is document["passed"] is document["sources_unchanged"] is True,
            f"Closed independently replayed phase required: {name}",
        )
    prefix.require(
        refs["baseline_branch"] in documents["baseline_producer"]["branch_refs"]
        and refs["baseline_branch_checker"] in documents["baseline_checker"]["branch_refs"],
        "Fixed baseline branches must belong to their closed phases",
    )
    baseline, checked, shadow = (documents[key] for key in ("baseline_branch", "baseline_branch_checker", "shadow"))
    prefix.require(
        baseline["complete"] is baseline["passed"] is checked["complete"] is checked["passed"] is True
        and baseline["case"] == checked["case"]
        and baseline["branch"] == checked["branch"] == "latest"
        and len(baseline["runtime"]["continuation"]["periods"])
        == len(checked["runtime"]["continuation"]["periods"])
        == 8
        and shadow["complete"] is shadow["passed"] is True
        and shadow["studied_steps"] == 8
        and shadow["studied_deferred"] == 1,
        "Complete baseline, independent replay and studied shadow required",
    )
    studied_ref = shadow["case_refs"][-1]
    studied = prefix.read(studied_ref)
    prefix.require(
        studied["case_id"] == "task1_init0_s43_terminal058_latest"
        and studied["deferred"] == len(studied["triggers"]) == 1
        and studied["triggers"][0]["step"] == 234
        and studied["triggers"][0]["baseline_command"] == 1.0
        and studied["triggers"][0]["proposed_command"] == 0.0,
        "Fixed previously studied trigger required",
    )
    for left, right in zip(
        baseline["runtime"]["continuation"]["periods"], checked["runtime"]["continuation"]["periods"], strict=True
    ):
        np.testing.assert_array_equal(left["command"], right["command"])
        np.testing.assert_allclose(left["actual_state"], right["actual_state"], atol=1e-10, rtol=0)
    prior = earlier.load_inputs()
    case = next(item for item in prior["cases"] if item["spec"]["case_index"] == 7)
    selected = earlier.selected_case(case, "latest")
    prefix.require(
        case["spec"] == baseline["case"]
        and case["spec"]["case_id"] == "task1_init0_s43_terminal058"
        and case["spec"]["prefix_steps"] == 232
        and case["config"]["seed"] == 43
        and case["config"]["task_id"] == 9
        and selected["cursor"] == 0,
        "Original runtime, history and already-generated reference required",
    )
    consumed = dict(prior["source_sha256"])
    for ref in (config_ref, *refs.values(), studied_ref):
        prefix.checked(ref, consumed)
    return dict(
        config_ref=config_ref,
        config=config,
        refs=refs,
        studied_ref=studied_ref,
        case=case,
        selected=selected,
        baseline=baseline,
        asset_refs=prior["asset_refs"],
        cv2_admission=prior["cv2_admission"],
        source_sha256=consumed,
    )


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "refs": inputs["refs"],
        "studied_ref": inputs["studied_ref"],
        "spec": inputs["case"]["spec"],
        "asset_refs": inputs["asset_refs"],
        "cv2_admission": inputs["cv2_admission"],
        "source_sha256": inputs["source_sha256"],
    }
