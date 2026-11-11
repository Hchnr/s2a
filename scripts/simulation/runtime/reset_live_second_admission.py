"""Freeze inputs and source identities for one reset-to-live-WAM continuation."""

import hashlib
import json
from pathlib import Path

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_segmented_reset_equivalence as segmented
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_live_second_window_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_live_second_window_20261111.md"
OUTPUT = ROOT / "outputs/reset_live_second_window_20261111"
QA = ROOT / "outputs/qa/reset_live_second_window_20261111/result.json"
DRY = ROOT / "outputs/reset_live_second_window_dry_20261111/result.json"
GPU = "GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf"
SESSION = "reset_case8_live_s42_20261111"
BASELINE = {
    "path": str(ROOT / "outputs/segmented_reset_equivalence_20261111/producer/result.json"),
    "sha256": "182f91a7603c94b58a9c0bd832773cbe8346a909517c4bda178b8ccf6c68d150",
}
AUDIT = {
    "path": str(ROOT / "outputs/segmented_reset_equivalence_audit_20261111/result.json"),
    "sha256": "d5f5da6635b6c2c297f38614b490fd539abec6ac45865079087b71935eb42066",
}
FIXED = {
    "schema": "s2a.reset-live-second-window.v1",
    "output": str(OUTPUT.relative_to(ROOT)),
    "case_index": 8,
    "seed": 42,
    "task": "put the cream cheese in the bowl",
    "gpu_uuid": GPU,
    "first_window_steps": 8,
    "live_window_steps": 8,
    "max_model_initializations": 1,
    "max_model_requests": 1,
    "denoise_steps": 35,
    "max_control_steps_per_phase": 16,
    "max_native_steps_per_phase": 400,
    "max_models_per_phase": 3,
    "max_contexts_per_phase": 3,
    "max_quality_responses_per_period": 320,
    "max_baseline_responses_per_period": 480,
    "max_gripper_sequences_per_period": 6561,
    "max_audit_endpoint_queries": 3200,
    "max_audit_transition_steps": 41600,
    "model_sha256": "48138bb8c95ce8cc1ca0b70f925e57c8f16bfb21be0a243495b391a06cf5b8ec",
    "worker_timeout_seconds": 2400,
    "world_model_seed": 42,
    "fresh_cases": 0,
    "fits": 0,
    "automatic_retry": False,
}


def validate_config(config):
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact reset case, device, model and physical budgets are required",
    )


def source_refs():
    names = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/run_reset_live_second_window.py",
        ROOT / "scripts/simulation/runtime/reset_live_two_window.py",
        ROOT / "scripts/simulation/runtime/reset_live_two_window_replay.py",
        ROOT / "scripts/wam/tracking/preview/reset_live_request.py",
        ROOT / "src/s2a/research/native_runtime/reset_two_window_runtime.py",
        ROOT / "src/s2a/research/native_runtime/reset_stream_boundary.py",
        ROOT / "src/s2a/research/native_runtime/streaming_continuation.py",
        ROOT / "src/s2a/research/native_runtime/endpoint_cache.py",
        ROOT / "src/s2a/research/wam/tracking/runtime/causal_stream_reference.py",
        ROOT / "src/s2a/research/wam/tracking/runtime/causal_stream_protocol.py",
        ROOT / "src/s2a/infrastructure/lifecycle/owned_socket_pair.py",
        ROOT / "scripts/wam/tracking/preview/live_cv2_admission.py",
        ROOT / "scripts/wam/tracking/preview/ordered_feedback_wam_inference_20261111.py",
        ROOT / "scripts/wam/tracking/preview/gpu_resume_repro_20261111.py",
        ROOT / "src/s2a/backends/wam/runtime.py",
        ROOT / "scripts/wam/readout/run_wam_readout_features.py",
        ROOT / "scripts/wam/training/clean/qualify_wam_clean_bridge.py",
        ROOT / "tests/research/native_runtime/test_reset_live_second_admission.py",
        ROOT / "tests/research/native_runtime/test_reset_live_second_owner.py",
        ROOT / "tests/research/native_runtime/test_reset_live_two_window.py",
        ROOT / "tests/research/native_runtime/test_reset_live_two_window_replay.py",
        ROOT / "tests/research/native_runtime/test_reset_two_window_runtime.py",
        ROOT / "tests/research/wam/test_reset_live_request.py",
    )
    return segmented.source_refs() + [prefix.reference(path) for path in names]


def load_inputs(*, verify_assets=True):
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    inputs = segmented.load_inputs(verify_assets=verify_assets)
    baseline, audit = prefix.read(BASELINE), prefix.read(AUDIT)
    prefix.require(
        baseline["complete"] is baseline["passed"] is baseline["sources_unchanged"] is True
        and baseline["counts"]["totals"]["control_steps"] == 96
        and audit["complete"] is audit["passed"] is audit["sources_unchanged"] is True
        and audit["producer_ref"] == BASELINE
        and audit["checker_candidate_modules_unloaded"] == 12
        and audit["original_commands_compared"] == audit["checker_commands_compared"] == 96,
        "The closed, independently replayed segmented reset baseline is required",
    )
    branch_ref = baseline["branch_refs"][config["case_index"]]
    branch = prefix.read(branch_ref)
    case = inputs["cases"][config["case_index"]]
    periods = branch["runtime"]["continuation"]["periods"]
    prefix.require(
        branch["complete"] is branch["passed"] is True
        and branch["case"] == case["case"]
        and case["case"]["case_id"] == "task2_init0_s42_raw"
        and len(periods) == 8
        and branch["runtime"]["continuation"]["stop"] is None,
        "The fixed raw task2 reset and all eight observed periods are required",
    )
    old_config = prefix.read(prefix.reference(repro.CONFIG))
    repro.validate_config(old_config)
    prepared, bridge, _completed, template, _response = engine.previous_inputs()
    model_config = prepared["model_config"]
    cosmos = Path(model_config["cosmos_source_root"]).resolve()
    prefix.require(
        cosmos == Path(bridge["config"]["cosmos_source_root"]).resolve()
        and cosmos.is_relative_to(ROOT / "outputs")
        and Path(model_config["model_python"]).is_file()
        and model_config["checkpoint"]["name"] == "8400_masked"
        and model_config["checkpoint"]["iteration"] == "iter_000008400",
        "The fixed read-only model Python, source and checkpoint are required",
    )
    sources = {path: digest for path, digest in bridge["source_sha256"].items() if Path(path).is_relative_to(cosmos)}
    weights = {path: detail["sha256"] for path, detail in prepared["weight_files"].items()}
    fingerprint = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    prefix.require(
        len(sources) >= 100 and len(weights) == 163 and fingerprint == config["model_sha256"],
        "The checkpoint and source manifests must match",
    )
    refs = dict(
        config_ref=config_ref,
        baseline=BASELINE,
        audit=AUDIT,
        baseline_branch=branch_ref,
        repro_config=prefix.reference(repro.CONFIG),
        prepared=dict(path=str(ROOT / old_config["prepared"]), sha256=old_config["prepared_sha256"]),
    )
    consumed = dict(inputs["source_sha256"])
    for ref in refs.values():
        prefix.checked(ref, consumed)
    end = periods[-1]["endpoint_receipt"]
    initial = {"step": end["endpoint_step"], "sim_time_s": end["sim_time"], "state_sha256": end["sim_state_sha256"]}
    prefix.require(initial["step"] == 8, "The model must begin at the completed current endpoint")
    return dict(
        base=inputs,
        config=config,
        refs=refs,
        case=case,
        baseline_periods=periods,
        initial=initial,
        prepared=prepared,
        bridge=bridge,
        template=template,
        model_source_hashes=sources,
        weight_hashes=weights,
        source_sha256=consumed,
    )


def input_receipt(inputs):
    return dict(
        refs=inputs["refs"],
        case=inputs["case"]["case"],
        initial=inputs["initial"],
        source_sha256=inputs["source_sha256"],
    )


def model_registration(inputs, image_root):
    return dict(
        session=SESSION,
        initial=inputs["initial"],
        image_root=str(Path(image_root).relative_to(ROOT)),
        task=inputs["config"]["task"],
        seed=inputs["config"]["world_model_seed"],
        model_sha256=inputs["config"]["model_sha256"],
        max_requests=1,
        denoise_steps=35,
    )
