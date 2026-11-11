"""Independently score two retained WAM responses without rerunning the model."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from s2a.geometry import pose_error
from s2a.references.action_reference import decode_action_targets
from scripts.simulation.runtime import prepare_ordered_correct_stream as binding
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = repro.ROOT
CONFIG = ROOT / "configs/tracking/references/ordered_feedback_wam_partial_audit_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/ordered_feedback_wam_partial_audit_20261111.md"
ORIGINAL_CONFIG = ROOT / "configs/tracking/references/ordered_feedback_wam_inference_20261111.json"
OUTPUT = ROOT / "outputs/ordered_feedback_wam_observation_v2_20261111/inference_audit"
INFERENCE = OUTPUT.parent / "inference"
QA = ROOT / "outputs/qa/ordered_feedback_wam_partial_audit_20261111/result.json"
TASK = "put the cream cheese in the bowl"
RUNNER = "scripts/wam/tracking/preview/ordered_feedback_wam_inference_20261111.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_ref(config, key):
    return repro.read_checked(ROOT / config[key], config[key + "_sha256"])


def validate_config(config):
    fixed = {
        "schema": "s2a.ordered-feedback-wam-partial-audit.v1",
        "output": str(OUTPUT.relative_to(ROOT)),
        "frozen_runner_commit": "e19af19b70744efbb440f37320837015f8d9b638",
        "frozen_runner_sha256": "3e1eaae714fb913d114e062ae3362f482e35be1a5386c3b2d05cdef1a7276468",
        "previous_model_initializations": 1,
        "previous_requests": 2,
        "previous_denoise_calls": 70,
        "new_model_initializations": 0,
        "new_model_requests": 0,
        "new_physics_steps": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    require(
        all(config.get(key) == value and type(config.get(key)) is type(value) for key, value in fixed.items())
        and [row["step"] for row in config["responses"]] == [80, 88],
        "Fixed no-new-model partial audit required",
    )


def source_refs():
    paths = (
        CONFIG,
        PLAN,
        ORIGINAL_CONFIG,
        Path(__file__).resolve(),
        ROOT / "tests/research/wam/test_ordered_feedback_wam_partial_audit.py",
        ROOT / "src/s2a/references/action_reference.py",
        ROOT / "src/s2a/geometry/__init__.py",
    )
    return {str(path.relative_to(ROOT)): repro.sha256(path) for path in paths}


def validate_request_response(request, response, conditioning, image, runtime):
    require(
        set(request) == repro.EXPECTED_KEYS and set(request["extra"]) == {"action128"},
        "Unexpected model request conditioning",
    )
    extra = request["extra"]["action128"]
    require(
        set(extra) == repro.EXTRA_KEYS
        and request["name"] == f"s2a_correct08_t{image['step']}_s42"
        and request["vision_path"] == str(ROOT / image["path"])
        and request["prompt"] == extra["task_title"] == TASK
        and [request[k] for k in ("image_size", "action_chunk_size", "fps", "num_frames", "seed", "num_steps")]
        == [960, 40, 20, 41, 42, 35]
        and request["model_mode"] == "wam"
        and request["domain_name"] == "franka_panda__parallel_gripper"
        and request["guidance"] == 1.0
        and request["shift"] == 5.0
        and extra["action_fps"] == 20
        and extra["action_dim_mask"] == [True] * 10 + [False] * 118
        and all(type(value) is bool for value in extra["action_dim_mask"])
        and extra["selected_roles"] == ["auxiliary_0", "left_wrist"]
        and extra["camera_layout"] == [["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]]
        and extra["return_video"] is False,
        "Request must contain only the registered current image and task",
    )
    caption = conditioning["caption"]
    require(
        conditioning["image_sha256"] == image["sha256"]
        and caption["task_title"] == caption["command"] == TASK
        and caption["task_plan"] == []
        and caption["camera_layout"] == extra["camera_layout"],
        "Saved actual conditioning differs",
    )
    args = response["args"]
    require(
        response["status"] == "success"
        and len(response["outputs"]) == 1
        and all(
            args[key] == request[key]
            for key in ("name", "vision_path", "prompt", "seed", "num_steps", "action_chunk_size", "fps", "num_frames")
        )
        and args["extra"] == request["extra"],
        "Saved response does not identify the registered request",
    )
    content = response["outputs"][0]["content"]
    metadata = content["action128"]
    require(
        metadata["action_domain"] == extra["action_domain"]
        and metadata["action_dim_mask"] == extra["action_dim_mask"]
        and metadata["action_fps"] == 20
        and metadata["seed"] == 42
        and metadata["checkpoint_path"] == runtime["checkpoint_path"]
        and np.allclose(metadata["action_timestamps_seconds"], np.arange(40) * 0.05, atol=1e-12, rtol=0),
        "Response domain, mask, timing, or checkpoint differs",
    )
    arrays = [np.asarray(content["action"], dtype=float), np.asarray(metadata["normalized_action"], dtype=float)]
    require(
        all(array.shape == (40, 128) and np.isfinite(array).all() and not np.any(array[:, 10:]) for array in arrays),
        "Both saved Action128 arrays must be finite H40 with 118 zero inactive slots",
    )
    return arrays[0]


def score_saved_response(anchor, action, targets, openings):
    prediction, predicted_opening = decode_action_targets(anchor, action, horizon=40)
    delta = pose_error(targets, prediction)
    result = {}
    for horizon in (8, 40):
        error = delta[:horizon]
        opening_error = predicted_opening[:horizon] - openings[:horizon]
        result[f"h{horizon}"] = {
            "position_rmse_m": float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
            "position_max_m": float(np.max(np.linalg.norm(error[:, :3], axis=1))),
            "rotation_rmse_rad": float(np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1)))),
            "rotation_max_rad": float(np.max(np.linalg.norm(error[:, 3:], axis=1))),
            "opening_rmse": float(np.sqrt(np.mean(opening_error**2))),
            "opening_max_absolute": float(np.max(np.abs(opening_error))),
        }
    return result


def load_audit(config):
    preflight, failed, owner, runtime = (read_ref(config, key) for key in ("preflight", "failed", "owner", "runtime"))
    require(repro.sha256(ROOT / config["worker_log"]) == config["worker_log_sha256"], "Worker log changed")
    frozen = subprocess.check_output(["git", "show", f"{config['frozen_runner_commit']}:{RUNNER}"], cwd=ROOT)
    require(
        hashlib.sha256(frozen).hexdigest() == config["frozen_runner_sha256"]
        and preflight["local_sources"][RUNNER] == config["frozen_runner_sha256"]
        and b"len(evidence) == 1" in frozen
        and preflight["complete"] is True
        and preflight["config_sha256"] == repro.sha256(ORIGINAL_CONFIG),
        "Original runner, its one-evidence check, or source preflight changed",
    )
    require(
        failed["complete"] is False
        and failed["model_initializations"] == config["previous_model_initializations"]
        and failed["model_requests"] == config["previous_requests"]
        and failed["denoise_calls_attempted"] == failed["denoise_calls_completed"] == config["previous_denoise_calls"]
        and failed["error"] == "ValueError('Denoise or active-mask evidence differs')"
        and owner["complete"] is False
        and owner["actual_wait"] is owner["reaped"] is owner["released"] is True
        and owner["worker_exit_code"] == 1
        and owner["signal_sent"] is False
        and owner["after"]["memory_mib"] == 0
        and owner["after"]["compute_apps"] == []
        and runtime["ema_weights"] is False,
        "Failed postflight and actual released-worker evidence required",
    )
    original = json.loads(ORIGINAL_CONFIG.read_text())
    summary = repro.read_checked(ROOT / original["capture_summary"], original["capture_summary_sha256"])
    branch = repro.read_checked(ROOT / original["capture_branch"], original["capture_branch_sha256"])
    captures = branch["runtime"]["captures"]
    require(
        summary["complete"] is summary["passed"] is branch["complete"] is branch["passed"] is True
        and [capture["frame_index"] for capture in captures] == [80, 88]
        and len(failed["completed_responses"]) == len(config["responses"]) == len(captures) == 2,
        "Two completed physical capture endpoints and saved responses are required",
    )
    inputs = binding.load_inputs()
    require(inputs["previous"]["record"]["task"] == TASK, "Correct-reference task identity differs")
    rows = []
    for index, (image, capture, expected, saved) in enumerate(
        zip(original["images"], captures, config["responses"], failed["completed_responses"], strict=True)
    ):
        require(
            image["step"] == capture["frame_index"] == expected["step"] == saved["step"]
            and saved["request_sha256"] == expected["request_sha256"]
            and saved["response_sha256"] == expected["response_sha256"]
            and capture["canvas_png"] == {"path": str(ROOT / image["path"]), "sha256": image["sha256"]}
            and capture["render_before"] == capture["render_after"]
            and repro.sha256(ROOT / image["path"]) == image["sha256"],
            "Image, endpoint, and retained response bindings differ",
        )
        directory = INFERENCE / f"model_run/step{image['step']}"
        request = repro.read_checked(directory / "request.json", expected["request_sha256"])
        response = repro.read_checked(directory / "response.json", expected["response_sha256"])
        conditioning = repro.read_checked(directory / "conditioning.json", expected["conditioning_sha256"])
        action = validate_request_response(request, response, conditioning, image, runtime)
        anchor = np.asarray(capture["current_source_pose"]) @ inputs["previous"]["tool"]
        rows.append(
            {
                "step": image["step"],
                "response_sha256": expected["response_sha256"],
                "elapsed_seconds": saved["elapsed_seconds"],
                "errors": score_saved_response(anchor, action, inputs["targets"][index], inputs["openings"][index]),
            }
        )
    return {
        "complete": True,
        "original_worker_success": False,
        "mask_initial_noise_ledger_available": False,
        "producer_self_reported_denoise_attempted": failed["denoise_calls_attempted"],
        "producer_self_reported_denoise_completed": failed["denoise_calls_completed"],
        "rows": rows,
        "model_inputs_included_future_truth": False,
        "physical_closed_loop": False,
        "new_model_requests": 0,
        "new_physics_steps": 0,
        "fresh_cases": 0,
        "source_sha256": inputs["source_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", required=True)
    parser.parse_args()
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("LD_LIBRARY_PATH") == ""
        and Path.cwd().resolve() == ROOT,
        "CPU-only repository environment required",
    )
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    qa = json.loads(QA.read_text())
    snapshot_ref = qa["source_before_ref"]
    snapshot = repro.read_checked(snapshot_ref["path"], snapshot_ref["sha256"])
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and qa["skipped_count"] == 0
        and all(snapshot["sources"].get(path) == digest for path, digest in source_refs().items()),
        "Passing CPU audit regression on current sources required",
    )
    result = load_audit(config)
    result["qa_sha256"] = repro.sha256(QA)
    result["audit_sources"] = source_refs()
    OUTPUT.mkdir(exist_ok=False)
    repro.write_new(OUTPUT / "result.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "source_sha256"}), flush=True)


if __name__ == "__main__":
    main()
