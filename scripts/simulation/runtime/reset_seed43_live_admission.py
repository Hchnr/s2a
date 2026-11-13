"""Admit one seed-43 reset-to-current-image development continuation."""

import hashlib
import json
from pathlib import Path

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_segmented_reset_equivalence as segmented
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_seed43_live_second_window_20261113.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_seed43_live_second_window_20261113.md"
OUTPUT = ROOT / "outputs/reset_seed43_live_second_window_20261113"
QA = ROOT / "outputs/qa/reset_seed43_live_second_window_final_20261113/result.json"
DRY = ROOT / "outputs/reset_seed43_live_second_window_dry_20261113/result.json"
CONFIG_SHA = "60781b748862164715ab0567b92a2c89ce406e8ae4a01688702715fb45696c36"
PLAN_SHA = "3256bd9371793b8668dbae46f633436b90a71f07ce4e832de9899b6316cbe707"
SESSION = "reset_case2_live_s43_20261113"
GPU = "GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf"


def _same(value, expected):
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return value.keys() == expected.keys() and all(_same(value[key], item) for key, item in expected.items())
    if type(expected) is list:
        return len(value) == len(expected) and all(_same(a, b) for a, b in zip(value, expected, strict=True))
    return value == expected


def validate_config(config):
    """Require the exact preregistered bytes and strictly typed configuration."""
    registered = prefix.read({"path": str(CONFIG), "sha256": CONFIG_SHA})
    prefix.require(_same(config, registered), "The exact registered seed-43 reset and budgets are required")
    prefix.require(
        config["gpu_uuid"] == GPU
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and config["qa"] == str(QA.relative_to(ROOT))
        and config["dry"] == str(DRY.relative_to(ROOT)),
        "The local output paths and registered GPU must agree",
    )


def configured_ref(config, key):
    return {"path": str(ROOT / config[key]), "sha256": config[key + "_sha256"]}


def check_history(config, base, baseline, audit, branch, equivalence):
    """Bind the historical reset, audited eight periods and seed-43 reference."""
    index = config["case_index"]
    case = base["cases"][index]
    branch_ref = configured_ref(config, "historical_case_branch")
    baseline_ref = configured_ref(config, "historical_segmented_producer")
    historical = equivalence["historical_short_window"]
    prefix.require(
        baseline["complete"] is baseline["passed"] is baseline["sources_unchanged"] is True
        and baseline["counts"]["totals"]["control_steps"] == 96
        and len(baseline["branch_refs"]) == 12
        and baseline["branch_refs"][index] == branch_ref
        and audit["complete"] is audit["passed"] is audit["sources_unchanged"] is True
        and audit["producer_ref"] == baseline_ref
        and audit["checker_candidate_modules_unloaded"] == 12
        and audit["original_commands_compared"] == audit["checker_commands_compared"] == 96
        and audit["cases"][index]["producer_ref"] == branch_ref
        and equivalence["complete"] is equivalence["passed"] is True
        and equivalence["evidence_level"] == "retrospective_reference_equivalence_not_new_closed_loop"
        and audit["cases"][index]["original_ref"] == equivalence["historical_producer_ref"]
        and historical["control_periods"] == 8
        and set(historical["replay_max_absolute_difference"])
        == {"command", "actual_state", "tcp_error", "inner_time", "inner_q", "inner_qvel"}
        and all(value == 0 for value in historical["replay_max_absolute_difference"].values()),
        "The independently audited historical reset and reference equivalence are required",
    )
    periods = branch["runtime"]["continuation"]["periods"]
    reset = branch["runtime"]["reset_endpoint_comparison"]
    prefix.require(
        branch["complete"] is branch["passed"] is True
        and branch["case"] == case["case"]
        and case["case"]["case_index"] == index == 2
        and case["case"]["case_id"] == config["case_id"]
        and case["case"]["sampling_seed"] == config["sampling_seed"] == 43
        and case["target_object"] == "akita_black_bowl_1"
        and case["chunk"]["response_identity"]["seed"] == 43
        and case["chunk"]["response_sha256"] == reset["response_sha256"]
        and case["expected_receipt"]["sim_state_sha256"]
        == reset["recorded_state_sha256"]
        == reset["actual_state_sha256"]
        == config["initial_state_sha256"]
        and reset["state_bytes_equal"] is True
        and branch["runtime"]["reset_image_match"] is True
        and len(periods) == config["first_window_steps"] == 8
        and [row["step"] for row in periods] == list(range(8))
        and branch["runtime"]["continuation"]["stop"] is None,
        "The same raw seed-43 development reset and eight completed periods are required",
    )
    endpoint = periods[-1]["endpoint_receipt"]
    initial = dict(
        step=endpoint["endpoint_step"], sim_time_s=endpoint["sim_time"], state_sha256=endpoint["sim_state_sha256"]
    )
    prefix.require(
        initial["step"] == 8 and initial["state_sha256"] == config["historical_step8_state_sha256"],
        "The model request must bind the recorded step-8 state",
    )
    return case, periods, initial


def load_inputs(*, verify_assets=True):
    """Load the recorded reset and immutable WAM/checkpoint rosters on CPU."""
    prefix.environment()
    config_ref = {"path": str(CONFIG), "sha256": CONFIG_SHA}
    plan_ref = {"path": str(PLAN), "sha256": PLAN_SHA}
    config = prefix.read(config_ref)
    prefix.checked(plan_ref)
    validate_config(config)
    base = segmented.load_inputs(verify_assets=verify_assets)
    names = (
        "historical_segmented_producer",
        "historical_segmented_audit",
        "historical_case_branch",
        "reference_equivalence_audit",
    )
    refs = {name: configured_ref(config, name) for name in names}
    baseline, audit, branch, equivalence = (prefix.read(refs[name]) for name in names)
    case, periods, initial = check_history(config, base, baseline, audit, branch, equivalence)
    old_config_ref = prefix.reference(repro.CONFIG)
    old_config = prefix.read(old_config_ref)
    repro.validate_config(old_config)
    prepared, bridge, completed, template, _response = engine.previous_inputs()
    model_config = prepared["model_config"]
    cosmos = Path(model_config["cosmos_source_root"]).resolve()
    sources = {path: digest for path, digest in bridge["source_sha256"].items() if Path(path).is_relative_to(cosmos)}
    weights = {path: detail["sha256"] for path, detail in prepared["weight_files"].items()}
    fingerprint = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    prefix.require(
        cosmos.is_relative_to(ROOT / "outputs")
        and cosmos == Path(bridge["config"]["cosmos_source_root"]).resolve()
        and Path(model_config["model_python"]).is_file()
        and model_config["checkpoint"]["name"] == "8400_masked"
        and model_config["checkpoint"]["iteration"] == "iter_000008400"
        and len(sources) == 1004
        and len(weights) == 163
        and fingerprint == config["model_sha256"]
        and completed["case_id"] == "task0_init0_s42_raw",
        "The fixed model Python, source and checkpoint manifests are required",
    )
    old_completed = prefix.read(equivalence["historical_v4_ref"])
    prefix.require(
        old_completed["case_id"] == config["case_id"]
        and old_completed["image_ref"]["sha256"] == config["initial_image_sha256"]
        and old_completed["response_ref"]["sha256"] == case["chunk"]["response_sha256"],
        "The initial image and first reference must match the audited seed-43 output",
    )
    prefix.checked(old_completed["image_ref"])
    prefix.checked(old_completed["response_ref"])
    refs.update(
        config=config_ref,
        plan=plan_ref,
        repro_config=old_config_ref,
        prepared={"path": str(ROOT / old_config["prepared"]), "sha256": old_config["prepared_sha256"]},
        historical_completed=equivalence["historical_v4_ref"],
        historical_image=old_completed["image_ref"],
        historical_response=old_completed["response_ref"],
    )
    consumed = dict(base["source_sha256"])
    for ref in refs.values():
        prefix.checked(ref, consumed)
    return dict(
        base=base,
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
    """Create the original causal ledger contract for the actual step-8 image."""
    image_root = Path(image_root).resolve()
    prefix.require(
        image_root == OUTPUT / "producer/simulation/runtime", "The producer's current-image root is required"
    )
    config = inputs["config"]
    return dict(
        session=SESSION,
        initial=dict(inputs["initial"]),
        image_root=str(image_root.relative_to(ROOT)),
        task=config["task"],
        seed=config["sampling_seed"],
        model_sha256=config["model_sha256"],
        max_requests=config["max_model_requests"],
        denoise_steps=config["denoise_calls_per_request"],
    )
