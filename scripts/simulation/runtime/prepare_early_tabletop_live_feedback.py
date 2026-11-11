"""Admit only each branch's own current-image WAM response for physical feedback."""

from pathlib import Path

from scripts.simulation.runtime import prepare_early_tabletop_intervention as early
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import early_tabletop_live_wam as model
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine
from scripts.wam.tracking.preview.live_stream_requests import _check_response_identity

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/early_tabletop_live_wam_feedback_20261111.json"
CONFIG_SHA = "a5e7722ae389d7d390ce00a5700a3048e5ee1ed828400d369401aa5a2c7fc92d"
PLAN = ROOT / "plans/v0.0.1/tracking/references/early_tabletop_intervention_live_wam_20261111.md"
OUTPUT = ROOT / "outputs/early_tabletop_intervention_live_wam_20261111/phase_b"
DRY = ROOT / "outputs/early_tabletop_live_wam_feedback_dry_20261111/result.json"
QA = ROOT / "outputs/qa/early_tabletop_live_wam_feedback_20261111/result.json"


def verify_branch_binding(spec, image, output, captured):
    mode = spec["mode"]
    prefix.require(
        output["mode"] == image["mode"] == mode
        and output["step"] == image["step"] == 120
        and output["response_sha256"] == spec["sha256"]
        and output["guard"]
        == {"mode": mode, "image_sha256": spec["image_sha256"], "state_sha256": spec["state_sha256"]}
        and image["sha256"] == spec["image_sha256"]
        and image["state_sha256"] == spec["state_sha256"]
        and captured["current_endpoint"]["sim_state_sha256"] == spec["state_sha256"],
        "Each WAM response must bind its own branch's current image and state",
    )


def load_inputs():
    inputs = early.load_inputs()
    consumed = dict(inputs["early_tabletop_intervention"]["source_sha256"])
    config = prefix.read({"path": str(CONFIG), "sha256": CONFIG_SHA}, consumed)
    prefix.require(
        config["schema"] == "s2a.early-tabletop-live-wam-feedback.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and (config["case_index"], config["seed"], config["branch_step"], config["model_step"], config["live_periods"])
        == (8, 42, 118, 120, 8)
        and config["modes"] == ["original", "pose_bounded"]
        and [row["mode"] for row in config["responses"]] == config["modes"]
        and (config["max_control_steps_per_phase"], config["max_native_steps_per_phase"], config["max_new_commands"])
        == (128, 3200, 16)
        and config["max_native_models"] == config["max_software_contexts"] == 12
        and config["new_model_initializations"] == config["new_model_requests"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The exact current-image response and physical feedback budgets are required",
    )
    names = ("capture_owner", "model_owner", "model_result", "model_preflight")
    refs = {key: {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]} for key in names}
    capture, owner, result, preflight = (prefix.read(refs[key], consumed) for key in names)
    prefix.require(
        capture["complete"] is capture["passed"] is capture["ready_for_model"] is True
        and owner["complete"] is owner["actual_wait"] is owner["reaped"] is owner["released"] is True
        and owner["worker_exit_code"] == 0
        and owner["signal_sent"] is False
        and result["complete"] is preflight["complete"] is True
        and result["gpu_uuid"] == preflight["gpu_uuid"] == model.GPU
        and (result["model_initializations"], result["model_requests"], result["denoise_calls"]) == (1, 2, 70)
        and len(result["denoise_evidence"]) == 70
        and all(row["completed"] for row in result["denoise_evidence"])
        and result["physics_steps"] == result["fresh_cases"] == 0,
        "Completed capture, WAM responses and a reaped released GPU worker are required",
    )
    engine.validate_mask_evidence(result["mask_evidence"], 2)
    model_config = model.load_config()
    captures = model.capture_inputs(model_config)
    prefix.require(
        model_config["capture_owner_sha256"] == refs["capture_owner"]["sha256"], "The WAM capture owner differs"
    )
    for path, digest in preflight["local_sources"].items():
        prefix.checked({"path": str(ROOT / path), "sha256": digest}, consumed)
    prepared, _bridge, _completed, _template, _old_response = engine.previous_inputs()
    branches = {}
    for spec, image, output, captured in zip(
        config["responses"], model_config["images"], result["responses"], captures, strict=True
    ):
        mode = spec["mode"]
        verify_branch_binding(spec, image, output, captured)
        response_ref = {"path": str(ROOT / spec["path"]), "sha256": spec["sha256"]}
        response = prefix.read(response_ref, consumed)
        request_path = Path(response_ref["path"]).parent / "request.json"
        request_ref = {"path": str(request_path), "sha256": output["request_sha256"]}
        request = prefix.read(request_ref, consumed)
        engine.validate_request(request, image)
        _check_response_identity(
            response, model_request=request, task=engine.TASK, checkpoint=prepared["model_config"]["checkpoint"]
        )
        arrays = repro.response_arrays(response)
        producer = prefix.read(capture["producer_refs"][mode], consumed)
        checker = prefix.read(capture["checker_refs"][mode], consumed)
        prefix.require(
            producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True,
            "A complete early producer and replay are required",
        )
        branches[mode] = {
            "producer": producer,
            "checker": checker,
            "image_ref": captured["canvas_png"],
            "state_sha256": spec["state_sha256"],
            "response_ref": response_ref,
            "request_ref": request_ref,
            "actions": arrays["action"],
        }
    inputs["early_tabletop_live_feedback"] = {
        "config": config,
        "refs": refs,
        "source_sha256": consumed,
        "branches": branches,
    }
    return inputs


def input_receipt(inputs):
    item = inputs["early_tabletop_live_feedback"]
    return {
        "prior": early.input_receipt(inputs),
        "config_ref": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "refs": item["refs"],
        "branches": {
            mode: {key: value[key] for key in ("image_ref", "state_sha256", "response_ref", "request_ref")}
            for mode, value in item["branches"].items()
        },
        "source_sha256": item["source_sha256"],
    }
