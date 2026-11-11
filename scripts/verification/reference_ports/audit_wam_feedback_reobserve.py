"""Independently audit the two causally linked current-image WAM responses."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from s2a.infrastructure.publication import publish_result as publish
from scripts.wam.tracking.preview import audit_ordered_feedback_wam_partial_20261111 as partial
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine
from scripts.wam.tracking.preview import wam_feedback_reobserve_20261111 as reobserve

ROOT = repro.ROOT
CONFIG = ROOT / "configs/tracking/references/wam_feedback_reobserve_audit_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/wam_feedback_reobserve_audit_20261111.md"
OUTPUT = ROOT / "outputs/wam_feedback_reobserve_audit_20261111"
QA = ROOT / "outputs/qa/wam_feedback_reobserve_audit_20261111/result.json"
MODEL = ROOT / "outputs/wam_feedback_reobserve_v2_20261111/model_run"


def read_ref(config, name):
    return repro.read_checked(ROOT / config[name], config[name + "_sha256"])


def validate_config(config):
    fixed = {
        "schema": "s2a.wam-feedback-reobserve-audit.v1",
        "output": str(OUTPUT.relative_to(ROOT)),
        "requests": 2,
        "denoise_calls": 70,
        "new_model_requests": 0,
        "new_physics_steps": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    engine.require(
        all(config.get(key) == value and type(config.get(key)) is type(value) for key, value in fixed.items()),
        "One fixed CPU-only reobservation audit required",
    )


def source_refs():
    paths = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        Path(reobserve.__file__).resolve(),
        Path(engine.__file__).resolve(),
        Path(partial.__file__).resolve(),
        ROOT / "tests/research/wam/test_wam_feedback_reobserve_audit.py",
    )
    return {str(path.relative_to(ROOT)): repro.sha256(path) for path in paths}


def validate_ledger(ledger, mask_evidence):
    expected = ["s2a_correct08_t80_s42"] * 35 + ["s2a_correct08_t88_s42"] * 35
    engine.require(
        len(ledger) == 70
        and [row["index"] for row in ledger] == list(range(70))
        and [row["label"] for row in ledger] == expected
        and all(row["completed"] is True for row in ledger),
        "Every registered denoise call must complete in request order",
    )
    engine.validate_mask_evidence(mask_evidence, 2)


def load_audit(config):
    preflight, model, owner, gate, response80, response88 = (
        read_ref(config, name) for name in ("preflight", "model_result", "owner", "gate", "response80", "response88")
    )
    run_config = json.loads(reobserve.CONFIG.read_text())
    reobserve.validate_config(run_config)
    captures = reobserve.capture_inputs(run_config)
    engine.require(
        preflight["complete"] is True
        and preflight["config_sha256"] == repro.sha256(reobserve.CONFIG)
        and preflight["source_files_verified"] == 1004
        and preflight["weight_files_verified"] == 163
        and all(repro.sha256(ROOT / path) == digest for path, digest in preflight["local_sources"].items())
        and model["complete"] is True
        and model["model_initializations"] == 1
        and model["model_requests"] == config["requests"]
        and model["denoise_calls"] == config["denoise_calls"]
        and model["physics_steps"] == model["fresh_cases"] == 0
        and model["automatic_retry"] is False
        and owner["complete"] is owner["actual_wait"] is owner["reaped"] is owner["released"] is True
        and owner["worker_exit_code"] == 0
        and owner["signal_sent"] is False
        and owner["gpu_uuid"] == model["gpu_uuid"] == run_config["gpu_uuid"]
        and owner["after"]["memory_mib"] == 0
        and owner["after"]["compute_apps"] == [],
        "Frozen preflight, completed model, actual worker exit and GPU release required",
    )
    validate_ledger(model["denoise_evidence"], model["mask_evidence"])
    responses = [response80, response88]
    records = model["responses"]
    engine.require(
        [row["step"] for row in records] == [80, 88]
        and [row["response_sha256"] for row in records] == [config["response80_sha256"], config["response88_sha256"]],
        "Exactly the two registered saved responses required",
    )
    runtime_path = MODEL / "model/runtime.json"
    runtime = json.loads(runtime_path.read_text())
    prepared, _bridge, _completed, _template, _response = engine.previous_inputs()
    checkpoint = prepared["model_config"]["checkpoint"]
    expected_checkpoint = (Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model").resolve()
    engine.require(
        Path(runtime["checkpoint_path"]).resolve() == expected_checkpoint and runtime["ema_weights"] is False,
        "Actual checkpoint identity differs",
    )
    rows = []
    for image, capture, record, response in zip(run_config["images"], captures, records, responses, strict=True):
        directory = MODEL / f"step{image['step']}"
        request = repro.read_checked(directory / "request.json", record["request_sha256"])
        conditioning_path = directory / "conditioning.json"
        conditioning = json.loads(conditioning_path.read_text())
        engine.validate_request(request, image)
        partial.validate_request_response(request, response, conditioning, image, runtime)
        caption_sha = hashlib.sha256(json.dumps(conditioning["caption"]).encode()).hexdigest()
        engine.require(
            caption_sha == response["outputs"][0]["content"]["action128"]["conditioning_caption_sha256"],
            "Saved caption must match the model response caption digest",
        )
        rows.append(
            {
                "step": image["step"],
                "image_sha256": image["sha256"],
                "sim_state_sha256": capture["current_endpoint"]["sim_state_sha256"],
                "response_sha256": record["response_sha256"],
                "request_sha256": record["request_sha256"],
                "conditioning_sha256": repro.sha256(conditioning_path),
                "elapsed_seconds": record["elapsed_seconds"],
            }
        )
    previous = reobserve.read_ref(run_config, "previous_response")
    comparison = repro.compare_responses(response80, previous)
    engine.require(
        gate["step"] == 80
        and gate["passed"] is True
        and gate["comparison"] == comparison
        and all(row["elementwise_equal"] for row in comparison.values())
        and records[0]["guard"] == gate
        and records[1]["guard"]["first_response_exact"] is True,
        "Step80 must be exactly the reference actually used before the step88 image",
    )
    return {
        "complete": True,
        "causal_first_reference_exact": True,
        "comparison": comparison,
        "rows": rows,
        "denoise_calls": len(model["denoise_evidence"]),
        "mask_evidence": model["mask_evidence"],
        "gpu_worker_exit": owner["worker_exit_code"],
        "gpu_released": owner["released"],
        "previous_attempt_failed": True,
        "earlier_partial_model_failed": True,
        "new_model_requests": 0,
        "new_physics_steps": 0,
        "fresh_cases": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", required=True)
    parser.parse_args()
    engine.require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("LD_LIBRARY_PATH") == ""
        and Path.cwd().resolve() == ROOT,
        "CPU-only repository audit required",
    )
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    qa = json.loads(QA.read_text())
    snapshot = repro.read_checked(qa["source_before_ref"]["path"], qa["source_before_ref"]["sha256"])
    engine.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and qa["skipped_count"] == 0,
        "Passing CPU audit regression required",
    )
    engine.validate_qa_snapshot(snapshot, source_refs())
    result = load_audit(config)
    result["qa_sha256"] = repro.sha256(QA)
    result["source_sha256"] = source_refs()
    OUTPUT.mkdir(exist_ok=False)
    print(publish(OUTPUT / "result.json", result), flush=True)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
