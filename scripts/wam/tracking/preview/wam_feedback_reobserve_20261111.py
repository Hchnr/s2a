"""Generate two current-image WAM references with an exact first-response gate."""

import argparse
import json
from pathlib import Path

from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine

ROOT = repro.ROOT
CONFIG = ROOT / "configs/tracking/references/wam_feedback_reobserve_v2_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/wam_feedback_reobserve_context_repair_20261111.md"
OUTPUT = ROOT / "outputs/wam_feedback_reobserve_v2_20261111"
QA = ROOT / "outputs/qa/wam_feedback_reobserve_v2_20261111/result.json"
GPU = repro.GPU
TASK = engine.TASK


def validate_config(config):
    fixed = {
        "schema": "s2a.wam-feedback-reobserve.v2",
        "output": str(OUTPUT.relative_to(ROOT)),
        "task": TASK,
        "gpu_uuid": GPU,
        "seed": 42,
        "horizon": 40,
        "denoise_steps_per_request": 35,
        "model_initializations": 1,
        "model_requests": 2,
        "denoise_calls": 70,
        "previous_attempt_initializations": 1,
        "previous_attempt_requests": 1,
        "previous_attempt_denoise_calls": 35,
        "physics_steps": 0,
        "fresh_cases": 0,
        "fits": 0,
        "automatic_retry": False,
    }
    engine.require(
        all(config.get(key) == value and type(config.get(key)) is type(value) for key, value in fixed.items())
        and [image["step"] for image in config["images"]] == [80, 88],
        "Fixed causally gated two-request budget required",
    )
    capture_root = ROOT / "outputs/wam_feedback_current_observation_v2_20261111/producer"
    for image in config["images"]:
        path = (ROOT / image["path"]).resolve()
        engine.require(
            path.is_relative_to(capture_root) and path.name == "current.png" and len(image["sha256"]) == 64,
            "Registered current feedback image required",
        )


def read_ref(config, name):
    return repro.read_checked(ROOT / config[name], config[name + "_sha256"])


def capture_inputs(config):
    summary, branch, qa, feedback, checker, audit, previous = (
        read_ref(config, key)
        for key in (
            "capture_summary",
            "capture_branch",
            "capture_qa",
            "feedback_branch",
            "feedback_checker",
            "partial_audit",
            "previous_response",
        )
    )
    branch_ref = {"path": str(ROOT / config["capture_branch"]), "sha256": config["capture_branch_sha256"]}
    counts = summary["counts"]
    engine.require(
        summary["complete"] is summary["passed"] is summary["sources_unchanged"] is True
        and summary["branch_refs"] == [branch_ref]
        and counts["control_steps"] == 88
        and counts["native_steps"] == 2200
        and counts["prefix_inner_samples"] == 2000
        and counts["continuation_steps"] == 8
        and counts["captured_steps"] == [80, 88]
        and counts["newly_selected_commands"] == 0
        and branch["complete"] is branch["passed"] is branch["candidate_modules_unloaded"] is True
        and branch["runtime"]["cleanup_complete"] is True
        and qa["complete"] is qa["passed"] is qa["actual_wait"] is qa["reaped"] is True
        and qa["skipped_count"] == 0
        and feedback["complete"] is feedback["passed"] is checker["complete"] is checker["passed"] is True
        and checker["counts"]["continuation_steps"] == 8
        and audit["complete"] is True
        and audit["original_worker_success"] is False
        and audit["mask_initial_noise_ledger_available"] is False
        and audit["rows"][0]["response_sha256"] == config["previous_response_sha256"],
        "Completed feedback, independent replay, and exact capture evidence required",
    )
    worker_ref = summary["workers_ref"]
    workers = repro.read_checked(worker_ref["path"], worker_ref["sha256"])["workers"]
    engine.require(
        len(workers) == 1
        and workers[0]["returncode"] == 0
        and workers[0]["actual_wait"] is workers[0]["reaped"] is True
        and workers[0]["signal_sent"] is False,
        "Actual capture worker must be reaped",
    )
    captures = branch["runtime"]["captures"]
    engine.require(len(captures) == 2, "Exactly two actual captured endpoints required")
    for image, capture in zip(config["images"], captures, strict=True):
        expected = {"path": str(ROOT / image["path"]), "sha256": image["sha256"]}
        engine.require(
            capture["frame_index"] == image["step"]
            and capture["canvas_png"] == expected
            and capture["canvas_shape"] == [960, 640, 3]
            and capture["camera_roles"] == ["auxiliary_0", "left_wrist"]
            and capture["future_rgb_frames"] == 0
            and capture["render_before"] == capture["render_after"]
            and capture["render_before"]["sim_state_sha256"] == capture["current_endpoint"]["sim_state_sha256"]
            and capture["render_before"]["sim_time"] == capture["current_endpoint"]["sim_time"]
            and repro.sha256(expected["path"]) == expected["sha256"]
            and repro.sha256(capture["raw_cameras"]["path"]) == capture["raw_cameras"]["sha256"],
            "Current image and unchanged actual endpoint required",
        )
    first = feedback["runtime"]["current_wam_image"]
    last = feedback["runtime"]["continuation"]["periods"][-1]["endpoint_receipt"]
    engine.require(
        captures[0]["current_endpoint"]["sim_state_sha256"] == first["current_endpoint"]["sim_state_sha256"]
        and captures[0]["canvas_rgb_sha256"] == first["canvas_rgb_sha256"]
        and captures[1]["current_endpoint"]["sim_state_sha256"] == last["sim_state_sha256"]
        and captures[1]["current_endpoint"]["sim_time"] == last["sim_time"]
        and repro.sha256(previous["args"]["vision_path"]) == config["images"][0]["sha256"]
        and previous["args"]["prompt"] == TASK,
        "Images must bind the original first request and its actual completed feedback endpoint",
    )
    repro.response_arrays(previous)
    failed, owner, attempted = (
        read_ref(config, key) for key in ("previous_failed", "previous_owner", "previous_attempt_response")
    )
    comparison = repro.compare_responses(attempted, previous)
    engine.require(
        failed["complete"] is False
        and failed["model_initializations"] == config["previous_attempt_initializations"]
        and failed["model_requests"] == config["previous_attempt_requests"]
        and failed["denoise_calls_attempted"]
        == failed["denoise_calls_completed"]
        == config["previous_attempt_denoise_calls"]
        and failed["completed_responses"][0]["response_sha256"] == config["previous_attempt_response_sha256"]
        and "DomainContext" in failed["error"]
        and "response_guard" in failed["error"]
        and owner["complete"] is False
        and owner["worker_exit_code"] == 1
        and owner["actual_wait"] is owner["reaped"] is owner["released"] is True
        and owner["signal_sent"] is False
        and owner["after"]["memory_mib"] == 0
        and owner["after"]["compute_apps"] == []
        and all(row["elementwise_equal"] for row in comparison.values()),
        "The failed prior attempt, its consumed budget, and its exact saved response must be verifiable",
    )
    engine.validate_mask_evidence(failed["mask_evidence"], 1)
    return captures


class FirstResponseGate:
    """Permit the second current-image request only after exact first-response reproduction."""

    def __init__(self, previous):
        repro.response_arrays(previous)
        self.previous = previous
        self.first_started = False
        self.first_passed = False
        self.second_started = False

    def before_request(self, image):
        if image["step"] == 80:
            engine.require(not self.first_started, "First request may be started only once")
            self.first_started = True
            return
        engine.require(
            image["step"] == 88 and self.first_passed and not self.second_started,
            "The second request requires an exact first-response reproduction",
        )
        self.second_started = True

    def __call__(self, image, response, directory):
        if image["step"] == 80:
            engine.require(self.first_started and not self.first_passed, "One started first request is required")
            comparison = repro.compare_responses(response, self.previous)
            passed = all(row["elementwise_equal"] for row in comparison.values())
            result = {"step": 80, "passed": passed, "comparison": comparison}
            repro.write_new(directory / "causal_match.json", result)
            self.first_passed = passed
            engine.require(passed, "Step80 response differs from the previously executed reference")
            return result
        engine.require(
            image["step"] == 88 and self.first_passed and self.second_started,
            "The second request requires an exact first-response reproduction",
        )
        return {"step": 88, "first_response_exact": True}


def context(config):
    previous = read_ref(config, "previous_response")
    gate = FirstResponseGate(previous)
    return engine.InferenceContext(
        config_path=CONFIG,
        plan_path=PLAN,
        output=OUTPUT,
        qa=QA,
        capture_loader=capture_inputs,
        additional_sources=(
            Path(__file__).resolve(),
            ROOT / "tests/research/wam/test_wam_feedback_reobserve.py",
            ROOT / "tests/research/wam/test_ordered_feedback_wam_inference.py",
            ROOT / "configs/tracking/references/wam_feedback_reobserve_20261111.json",
            ROOT / "plans/v0.0.1/tracking/references/wam_feedback_reobserve_20261111.md",
        ),
        worker_module="scripts.wam.tracking.preview.wam_feedback_reobserve_20261111",
        request_guard=gate.before_request,
        response_guard=gate,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "launch", "run"))
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    {"preflight": engine.preflight, "launch": engine.launch, "run": engine.run}[args.mode](config, context(config))


if __name__ == "__main__":
    main()
