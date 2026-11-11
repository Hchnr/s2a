"""Admit one source-pinned live WAM stream reproduction without launching it."""

from pathlib import Path

from scripts.simulation.runtime import causal_stream_recorded_admission as recorded
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/causal_stream_live_wam_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/causal_stream_live_wam_20261111.md"
OUTPUT = ROOT / "outputs/causal_stream_live_wam_20261111"
QA = ROOT / "outputs/qa/causal_stream_live_wam_20261111/result.json"
DRY = ROOT / "outputs/causal_stream_live_wam_dry_20261111/result.json"
GPU = "GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf"

FIXED = {
    "schema": "s2a.causal-stream-live-wam.v1",
    "output": "outputs/causal_stream_live_wam_20261111",
    "prior_stream_audit": "outputs/causal_stream_recorded_audit_20261111/result.json",
    "prior_stream_audit_sha256": "2647c5f9931ce4f5818fce07fe40142947acebc9c34134c2696efd5bbcff3bc4",
    "prior_stream_producer": "outputs/causal_stream_recorded_physical_20261111/producer/result.json",
    "prior_stream_producer_sha256": "43482cc62c975eca919ac51a57d68dbda1c3edc5e9652b1678dc09a0a9acd9e8",
    "prior_stream_checker": "outputs/causal_stream_recorded_physical_20261111/checker/result.json",
    "prior_stream_checker_sha256": "4ac014f9f560f06829c8a4f6b2f70f7330a50ba151962f116bb27ebdc1f7e805",
    "gpu_resume_config": "configs/tracking/gpu_resume_repro_20261111.json",
    "gpu_resume_config_sha256": "cd5ff9fc818109524b12be1204352ebe2bbe577c0a0eb7f4f3176cefc7986f05",
    "prepared": "outputs/tracking_preview_wam_development_v5_20261028/prepared.json",
    "prepared_sha256": "64ff281922755538537ec87e70e744788f04f82c4bd32a982d5033ace42525f7",
    "gpu_uuid": GPU,
    "case_index": 8,
    "seed": 42,
    "task": "put the cream cheese in the bowl",
    "prefix_steps": 80,
    "window_steps": 8,
    "windows": 2,
    "model_initializations": 1,
    "model_requests": 2,
    "denoise_calls_per_request": 35,
    "max_denoise_calls": 70,
    "max_control_steps_per_phase": 96,
    "max_native_steps_per_phase": 2400,
    "fresh_cases": 0,
    "fits": 0,
    "worker_timeout_seconds": 1800,
    "automatic_retry": False,
}


def validate_config(config):
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The live GPU, model and physical budgets must exactly match registration",
    )


def _ref(config, name):
    return {"path": str(ROOT / config[name]), "sha256": config[name + "_sha256"]}


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    inputs = recorded.load_inputs()
    consumed = dict(inputs["source_sha256"])
    names = ("prior_stream_audit", "prior_stream_producer", "prior_stream_checker", "gpu_resume_config", "prepared")
    refs = {name: _ref(config, name) for name in names}
    values = {name: prefix.read(ref, consumed) for name, ref in refs.items()}
    audit = values["prior_stream_audit"]
    prefix.require(
        audit["complete"] is audit["passed"] is audit["checker_policy_isolated"] is True
        and audit["control_qualified"] is False
        and audit["new_model_requests"] == audit["new_denoise_calls"] == 0
        and audit["replayed_prefix_inner_samples"] == 2000
        and audit["replayed_continuation_inner_samples"] == 400
        and audit["terminal_consecutive_success_periods"] == 9
        and audit["qualified_stable_success"] is False,
        "The recorded-reference stream and independent audit must be complete",
    )
    for name in ("prior_stream_producer", "prior_stream_checker"):
        value = values[name]
        prefix.require(
            value["complete"] is value["passed"] is value["sources_unchanged"] is True
            and value["counts"]["control_steps"] == 96
            and value["counts"]["native_steps"] == 2400
            and value["new_model_requests"] == value["new_denoise_calls"] == 0,
            "Both bounded prior physical phases must be closed",
        )
    old_config = values["gpu_resume_config"]
    repro.validate_config(old_config)
    prepared, bridge, _completed, template, _response = repro.inputs(old_config)
    prefix.require(
        prepared == values["prepared"] and len(prepared["weight_files"]) == 163,
        "The original complete checkpoint manifest is required",
    )
    model = prepared["model_config"]
    cosmos = Path(model["cosmos_source_root"]).resolve()
    prefix.require(
        cosmos == Path(bridge["config"]["cosmos_source_root"]).resolve()
        and cosmos.is_relative_to(ROOT / "outputs")
        and Path(model["model_python"]).is_file()
        and model["checkpoint"]["name"] == "8400_masked"
        and model["checkpoint"]["iteration"] == "iter_000008400",
        "The fixed read-only model Python, Cosmos snapshot and checkpoint are required",
    )
    sources = {path: digest for path, digest in bridge["source_sha256"].items() if Path(path).is_relative_to(cosmos)}
    weights = {path: detail["sha256"] for path, detail in prepared["weight_files"].items()}
    prefix.require(len(sources) >= 100 and len(weights) == 163, "Complete model source and checkpoint rosters required")
    for ref in (config_ref, *refs.values()):
        prefix.checked(ref, consumed)
    return {
        **inputs,
        "live_config_ref": config_ref,
        "live_config": config,
        "live_refs": refs,
        "prepared": prepared,
        "bridge": bridge,
        "template": template,
        "model_source_hashes": sources,
        "weight_hashes": weights,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "live_config_ref": inputs["live_config_ref"],
        "live_refs": inputs["live_refs"],
        "recorded_inputs": recorded.input_receipt(inputs),
        "model_sha256": inputs["model_sha256"],
        "source_sha256": inputs["source_sha256"],
    }
