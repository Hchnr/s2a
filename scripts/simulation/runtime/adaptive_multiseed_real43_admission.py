"""Admit one real seed-43 model replay on a saved development image."""

from pathlib import Path

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/adaptive_multiseed_real43_20261113.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/adaptive_multiseed_real43_20261113.md"
OUTPUT = ROOT / "outputs/adaptive_multiseed_real43_20261113"
QA = ROOT / "outputs/qa/adaptive_multiseed_real43_final_20261113/result.json"
DRY = ROOT / "outputs/adaptive_multiseed_real43_dry_20261113/result.json"
CONFIG_SHA = "2b891b4b8665f20a62feeb2fccfd44bdbff4f4043531852d3bab89bd35d6a982"
PLAN_SHA = "e64eaa5b93393fd8b28851b9ef65b52b7024747d7bcf049111d1b4c122904160"
GPU = "GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf"

FIXED = {
    "schema": "s2a.adaptive-multiseed-real43.v1",
    "output": str(OUTPUT.relative_to(ROOT)),
    "qa": str(QA.relative_to(ROOT)),
    "dry": str(DRY.relative_to(ROOT)),
    "session": "adaptive_multiseed_real43_saved_image_20261113",
    "gpu_uuid": GPU,
    "legacy_config": "configs/tracking/gpu_resume_repro_20261111.json",
    "legacy_config_sha256": "cd5ff9fc818109524b12be1204352ebe2bbe577c0a0eb7f4f3176cefc7986f05",
    "historical_case": "outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_raw/worker_result.json",
    "historical_case_sha256": "84ba26734fc3ee93fdda0e3fb99fece2f3f6c45793d76c0711d8b3ca5df8d17e",
    "historical_checkpoint_manifest": "outputs/wam_independent_20261020/checkpoint_fingerprints.json",
    "historical_checkpoint_manifest_sha256": "624ba2dd93b49fd76629f7a6c6905b2813b6605d48da8fc7c3f05a026da413a8",
    "historical_repeat_response": "outputs/gpu_resume_repro_20261111/model_run/response.json",
    "historical_repeat_response_sha256": "9669f30418803ac1f074b47d1fe9b132642fac2a4a0a37bd2509d2c603e9b45a",
    "model_sha256": "48138bb8c95ce8cc1ca0b70f925e57c8f16bfb21be0a243495b391a06cf5b8ec",
    "image_sha256": "17c5f04507ca8830657caa26f4ea5e18e3037309285a7752eaf4835d047e9a77",
    "initial_state_sha256": "5239d6a1c021e69231f3ef91a1ad93bd5b626981c003140b29150c0c9713a702",
    "initial_step": 0,
    "initial_sim_time_s": 0.25000000000000017,
    "task": "put the bowl on the plate",
    "seed": 43,
    "source_files": 1004,
    "weight_files": 163,
    "model_initializations": 1,
    "model_requests": 1,
    "denoise_calls": 35,
    "physics_steps": 0,
    "fresh_cases": 0,
    "worker_timeout_seconds": 1800,
    "automatic_retry": False,
}


def validate_config(config):
    """Freeze the complete one-request budget and historical identities."""
    prefix.require(
        type(config) is dict
        and config.keys() == FIXED.keys()
        and all(type(config[key]) is type(expected) and config[key] == expected for key, expected in FIXED.items()),
        "The registered replay input, UUID and exact budget must not change",
    )


def configured_ref(config, key):
    return {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]}


def load_inputs():
    """Rebind the saved image and seed-42 comparison without simulating."""
    from scripts.simulation.runtime import causal_stream_recorded_admission as fingerprint

    prefix.environment()
    config_ref = {"path": str(CONFIG), "sha256": CONFIG_SHA}
    plan_ref = {"path": str(PLAN), "sha256": PLAN_SHA}
    config = prefix.read(config_ref)
    prefix.checked(plan_ref)
    validate_config(config)
    refs = {
        key: configured_ref(config, key)
        for key in ("legacy_config", "historical_case", "historical_checkpoint_manifest", "historical_repeat_response")
    }
    legacy, case, checkpoint_manifest, repeated = (prefix.read(refs[key]) for key in refs)
    repro.validate_config(legacy)
    prepared, bridge, completed, template, previous = repro.inputs(legacy)
    seed42_comparison = repro.compare_responses(repeated, previous)
    prefix.require(
        all(item["elementwise_equal"] and item["different_elements"] == 0 for item in seed42_comparison.values()),
        "The retained seed-42 reproduction must match the historical arrays",
    )
    first = case["images"][0]
    endpoint = first["current_endpoint"]
    initial = dict(
        step=endpoint["endpoint_step"], sim_time_s=endpoint["sim_time"], state_sha256=endpoint["sim_state_sha256"]
    )
    expected_snapshot = dict(
        step=config["initial_step"],
        sim_time_s=config["initial_sim_time_s"],
        state_sha256=config["initial_state_sha256"],
    )
    rendered = {"sim_time": initial["sim_time_s"], "sim_state_sha256": initial["state_sha256"]}
    prefix.require(
        case["complete"] is case["passed"] is True
        and case["case"]["case_id"] == "task0_init0_s42_raw"
        and completed["case_id"] == case["case"]["case_id"]
        and first["frame_index"] == 0
        and initial == expected_snapshot
        and first["render_before"] == first["render_after"] == rendered
        and first["canvas_png"]["sha256"] == completed["image_ref"]["sha256"] == config["image_sha256"]
        and template["prompt"] == config["task"]
        and template["seed"] == completed["sampling_seed"] == 42,
        "The same saved development image, recorded endpoint and seed-42 task are required",
    )
    prefix.checked(first["canvas_png"])
    prefix.checked(completed["image_ref"])
    model_sha = fingerprint._fingerprint(checkpoint_manifest, config["model_sha256"])
    entries = [entry for entry in checkpoint_manifest["checkpoints"] if entry["name"] == "8400_masked"]
    weights = {path: value["sha256"] for path, value in prepared["weight_files"].items()}
    historical_weights = {path: value["sha256"] for path, value in entries[0]["files"].items()}
    root = Path(prepared["model_config"]["cosmos_source_root"]).resolve()
    model_sources = {path: sha for path, sha in bridge["source_sha256"].items() if Path(path).is_relative_to(root)}
    prefix.require(
        root.is_relative_to(ROOT / "outputs")
        and model_sha == config["model_sha256"]
        and weights == historical_weights
        and len(weights) == config["weight_files"]
        and len(model_sources) == config["source_files"],
        "The exact historical checkpoint and source rosters are required",
    )
    refs.update(
        config=config_ref,
        plan=plan_ref,
        prepared={"path": str(ROOT / legacy["prepared"]), "sha256": legacy["prepared_sha256"]},
        historical_completed={
            "path": str(ROOT / legacy["historical_completed"]),
            "sha256": legacy["historical_completed_sha256"],
        },
        historical_request=completed["request_ref"],
        historical_response=completed["response_ref"],
        historical_image=completed["image_ref"],
        recorded_canvas=first["canvas_png"],
        bridge=prepared["prerequisite_refs"]["bridge_prepared"],
    )
    for ref in refs.values():
        prefix.checked(ref)
    return dict(
        config=config,
        refs=refs,
        initial=initial,
        prepared=prepared,
        bridge=bridge,
        template=template,
        previous_response=previous,
        seed42_comparison=seed42_comparison,
        model_source_hashes=model_sources,
        weight_hashes=weights,
    )


def input_receipt(inputs):
    """Exclude unrelated historical labels and poses from the replay bundle."""
    return dict(
        refs=inputs["refs"],
        recorded_initial=inputs["initial"],
        model_sha256=inputs["config"]["model_sha256"],
        seed42_reproduction=inputs["seed42_comparison"],
        source_files=len(inputs["model_source_hashes"]),
        weight_files=len(inputs["weight_hashes"]),
        evidence_level="saved_development_image_model_replay",
    )


def model_registration(inputs, capture_root):
    """Register a replay frame; its snapshot is historical, not live state."""
    capture_root = Path(capture_root).resolve()
    prefix.require(capture_root == OUTPUT / "producer/simulation", "The registered replay image root is required")
    config = inputs["config"]
    return dict(
        session=config["session"],
        initial=dict(inputs["initial"]),
        image_root=str(capture_root.relative_to(ROOT)),
        task=config["task"],
        seed=config["seed"],
        model_sha256=config["model_sha256"],
        max_requests=config["model_requests"],
        max_periods=1,
        denoise_steps=config["denoise_calls"],
    )
