"""Admit a separately budgeted repair only after accounting for failed work."""

from scripts.simulation.runtime import prepare_early_tabletop_live_feedback as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/early_tabletop_live_feedback_repair_20261111.json"
CONFIG_SHA = "895b48fd77bf8e597a3b1f7d40330d0acc3d2809994fbfd812b92335eecfa2a3"
PLAN = ROOT / "plans/v0.0.1/tracking/references/early_tabletop_live_feedback_repair_20261111.md"
OUTPUT = ROOT / "outputs/early_tabletop_intervention_live_wam_20261111/phase_b_repair"
DRY = ROOT / "outputs/early_tabletop_live_feedback_repair_dry_20261111/result.json"
QA = ROOT / "outputs/qa/early_tabletop_live_feedback_repair_20261111/result.json"
ERROR = "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"


def validate_failed_branch(branch, *, mode, state_sha256):
    runtime = branch["runtime"]
    periods = runtime["continuation_partial"]["periods"]
    prefix.require(
        branch["complete"] is branch["passed"] is False
        and branch["mode"] == mode
        and branch["phase"] == "producer"
        and runtime["failure_stage"] == "continuation"
        and runtime["error"]["message"] == ERROR
        and runtime["cleanup_complete"] is True
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == 120
        and runtime["counters"]["mj_step"]["attempted"] == runtime["counters"]["mj_step"]["completed"] == 3000
        and runtime["counters"]["model_construct"]["completed"] == runtime["counters"]["context"]["completed"] == 3
        and runtime["newly_selected_commands"] == runtime["world_model_calls"] == runtime["fresh_cases"] == 0
        and [row["step"] for row in periods] == [118, 119]
        and periods[-1]["endpoint_receipt"]["sim_state_sha256"] == state_sha256,
        "The original producer must have stopped at its own step120 state before any new command",
    )


def load_inputs():
    inputs = previous.load_inputs()
    prior = inputs["early_tabletop_live_feedback"]
    consumed = dict(prior["source_sha256"])
    config = prefix.read({"path": str(CONFIG), "sha256": CONFIG_SHA}, consumed)
    prefix.require(
        config["schema"] == "s2a.early-tabletop-live-feedback-repair.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and config["case_index"] == 8
        and config["seed"] == 42
        and config["modes"] == ["original", "pose_bounded"]
        and (
            config["prior_capture_native_steps"],
            config["prior_failed_native_steps"],
            config["repair_max_native_steps"],
            config["total_max_native_steps"],
        )
        == (12000, 6000, 12800, 30800)
        and config["total_max_new_commands"] == 20
        and config["total_max_native_models"] == config["total_max_software_contexts"] == 30
        and config["new_model_initializations"] == config["new_model_requests"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The separately registered repair and cumulative budget are required",
    )
    names = ("failed_owner", "failed_workers", "failed_original", "failed_pose_bounded")
    refs = {key: {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]} for key in names}
    owner, workers, original, projected = (prefix.read(refs[key], consumed) for key in names)
    prefix.require(
        owner["complete"] is owner["passed"] is False
        and owner["error"]["message"] == "Every producers worker must exit and be reaped"
        and len(owner["groups"]) == 1
        and owner["groups"][0]["workers_ref"] == refs["failed_workers"]
        and workers["launch_error"] is None
        and len(workers["workers"]) == 2
        and all(
            row["actual_wait"] is row["reaped"] is True and row["returncode"] == 1 and row["signal_sent"] is False
            for row in workers["workers"]
        ),
        "Both failed workers must actually have exited and been reaped",
    )
    for mode, branch in zip(config["modes"], (original, projected), strict=True):
        expected = prior["branches"][mode]
        validate_failed_branch(branch, mode=mode, state_sha256=expected["state_sha256"])
        image = (
            ROOT
            / "outputs/early_tabletop_intervention_live_wam_20261111/phase_b"
            / mode
            / "producer/simulation/runtime/anchor0120/current.png"
        )
        prefix.checked({"path": str(image), "sha256": expected["image_ref"]["sha256"]}, consumed)
    inputs["early_tabletop_live_feedback_repair"] = {"config": config, "refs": refs, "source_sha256": consumed}
    return inputs


def input_receipt(inputs):
    item = inputs["early_tabletop_live_feedback_repair"]
    return {
        "prior": previous.input_receipt(inputs),
        "config_ref": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "refs": item["refs"],
        "source_sha256": item["source_sha256"],
    }
