"""Bind six reset cases to one adaptive WAM development protocol."""

import hashlib
import json
from pathlib import Path

from scripts.simulation.runtime import registered_adaptive_trial as registered
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as feedback
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine
from scripts.wam.tracking.preview.reset_multitask_live_requests import TASKS

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_multitask_adaptive_600_20261113.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_multitask_adaptive_600_20261113.md"
OUTPUT = ROOT / "outputs/reset_multitask_adaptive_600_20261113"
QA = ROOT / "outputs/qa/reset_multitask_adaptive_600_final_20261113/result.json"
DRY = ROOT / "outputs/reset_multitask_adaptive_600_dry_20261113"
CONFIG_SHA = "795ddb2f2ecaeae7669b8ba6e029798581f4b0d6eb0dc91e4d272714c1bb5616"
PLAN_SHA = "cf3101e738c9469a104d2ec678c72104ede706c95b57aa5d9c7424d2dbc15593"
BASELINE = {
    "path": str(ROOT / "outputs/segmented_reset_equivalence_20261111/producer/result.json"),
    "sha256": "182f91a7603c94b58a9c0bd832773cbe8346a909517c4bda178b8ccf6c68d150",
}
BASELINE_AUDIT = {
    "path": str(ROOT / "outputs/segmented_reset_equivalence_audit_20261111/result.json"),
    "sha256": "d5f5da6635b6c2c297f38614b490fd539abec6ac45865079087b71935eb42066",
}
MODULE = "scripts.simulation.runtime.run_reset_multitask_adaptive_600"
QA_TARGETS = (
    "tests/research/native_runtime/test_reset_multitask_adaptive_admission.py",
    "tests/research/native_runtime/test_reset_multitask_adaptive_execution.py",
    "tests/research/native_runtime/test_reset_multitask_adaptive_worker.py",
    "tests/research/native_runtime/test_reset_multitask_adaptive_matrix.py",
    "tests/research/native_runtime/test_reset_adaptive_audit.py",
    "tests/research/native_runtime/test_reset_adaptive_stream.py",
    "tests/research/native_runtime/test_adaptive_streaming_continuation.py",
    "tests/research/native_runtime/test_adaptive_streaming_replay.py",
    "tests/research/native_runtime/test_adaptive_stream_reference.py",
    "tests/research/native_runtime/test_registered_adaptive_execution.py",
    "tests/research/native_runtime/test_managed_physical_worker.py",
    "tests/research/native_runtime/test_physical_cost_bounds.py",
    "tests/unit/infrastructure/lifecycle/test_interruptible_worker.py",
    "tests/unit/infrastructure/lifecycle/test_owned_socket_pair.py",
    "tests/unit/infrastructure/test_adaptive_stream_protocol.py",
    "tests/research/wam/test_adaptive_multiseed_contract.py",
    "tests/research/wam/test_adaptive_multiseed_requests.py",
    "tests/research/wam/test_adaptive_multiseed_model_worker.py",
)


def config_ref():
    return {"path": str(CONFIG), "sha256": CONFIG_SHA}


def plan_ref():
    return {"path": str(PLAN), "sha256": PLAN_SHA}


def validate_config(config):
    """Require the registered roster, distinct devices per wave and full budgets."""
    expected = prefix.read(config_ref())
    prefix.require(type(config) is dict and config == expected, "The fixed six-case configuration is required")
    cases = config["cases"]
    indices = [0, 2, 4, 6, 8, 10]
    numeric = (
        "max_parallel_cases",
        "prefix_steps",
        "max_adaptive_periods",
        "max_control_steps_per_phase",
        "max_native_steps_per_phase",
        "max_models_per_phase",
        "max_contexts_per_phase",
        "max_model_initializations",
        "max_model_requests",
        "denoise_calls_per_request",
        "max_denoise_calls",
        "worker_timeout_seconds",
        "fresh_cases",
        "fits",
    )
    prefix.require(
        all(type(config[key]) is int for key in numeric)
        and all(type(index) is int for index in config["case_indices"])
        and config["case_indices"] == indices == [row["case_index"] for row in cases]
        and all(type(row["case_index"]) is int and type(row["sampling_seed"]) is int for row in cases)
        and [row["sampling_seed"] for row in cases] == [42, 43] * 3
        and all(row["case_id"] == f"task{row['case_index'] // 4}_init0_s{row['sampling_seed']}_raw" for row in cases)
        and len({row["gpu_uuid"] for row in cases[:4]}) == 4
        and len({row["gpu_uuid"] for row in cases[4:]}) == 2
        and all(type(row["gpu_uuid"]) is str and row["gpu_uuid"].startswith("GPU-") for row in cases)
        and config["max_parallel_cases"] == 4
        and config["prefix_steps"] == 8
        and config["max_adaptive_periods"] == 592
        and config["max_control_steps_per_phase"] == 600
        and config["max_native_steps_per_phase"] == 15000
        and config["max_models_per_phase"] == config["max_contexts_per_phase"] == 3
        and config["max_model_initializations"] == 1
        and config["max_model_requests"] == 75
        and config["denoise_calls_per_request"] == 35
        and config["max_denoise_calls"] == 2625
        and config["fresh_cases"] == config["fits"] == 0
        and config["automatic_retry"] is False,
        "The registered tasks, seeds, devices and physical/model budgets must agree",
    )


def case_config(config, index):
    """Create a per-case view without changing the shared registered protocol."""
    validate_config(config)
    prefix.require(type(index) is int and index in config["case_indices"], "One registered case index is required")
    spec = next(row for row in config["cases"] if row["case_index"] == index)
    case_dir = OUTPUT / f"case_{index:02d}"
    return dict(
        config,
        **spec,
        seed=spec["sampling_seed"],
        session=f"reset_multitask_adaptive_{index:02d}_20261113",
        output=str(case_dir.relative_to(ROOT)),
        qa=str(QA.relative_to(ROOT)),
        dry=str((DRY / f"case_{index:02d}.json").relative_to(ROOT)),
    )


def source_refs():
    """Freeze the shared execution, model and test sources for every case."""
    names = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/run_reset_multitask_adaptive_600.py",
        ROOT / "scripts/simulation/runtime/reset_multitask_adaptive_execution.py",
        ROOT / "scripts/simulation/runtime/reset_multitask_adaptive_worker.py",
        ROOT / "scripts/simulation/runtime/reset_adaptive_stream.py",
        ROOT / "scripts/simulation/runtime/reset_adaptive_audit.py",
        ROOT / "scripts/simulation/runtime/managed_physical_worker.py",
        ROOT / "scripts/simulation/runtime/registered_adaptive_trial.py",
        ROOT / "scripts/simulation/runtime/registered_adaptive_execution.py",
        ROOT / "scripts/simulation/runtime/registered_model_bundle.py",
        ROOT / "scripts/simulation/runtime/run_reset_start_feedback.py",
        ROOT / "scripts/simulation/runtime/run_segmented_reset_equivalence.py",
        ROOT / "scripts/wam/tracking/preview/adaptive_multiseed_contract.py",
        ROOT / "scripts/wam/tracking/preview/adaptive_multiseed_requests.py",
        ROOT / "scripts/wam/tracking/preview/adaptive_multiseed_model_worker.py",
        ROOT / "src/s2a/research/native_runtime/reset_bounded_runtime.py",
        ROOT / "src/s2a/research/native_runtime/adaptive_streaming_continuation.py",
        ROOT / "src/s2a/research/native_runtime/adaptive_streaming_replay.py",
        ROOT / "src/s2a/research/wam/tracking/runtime/adaptive_stream_protocol.py",
        ROOT / "src/s2a/research/wam/tracking/runtime/adaptive_stream_reference.py",
        ROOT / "src/s2a/research/joint_control/preview/causal_reference_replan.py",
        *(ROOT / path for path in QA_TARGETS),
    )
    return [prefix.reference(path) for path in names]


def load_inputs(index, *, verify_assets=True):
    """Admit the exact reset, independent first-window audit and model sources."""
    prefix.environment()
    config, _ = prefix.read(config_ref()), prefix.checked(plan_ref())
    current = case_config(config, index)
    base = feedback.load_inputs(index, verify_assets=verify_assets)
    prefix.require(len(base["cases"]) == 1, "One selected case is required")
    case = base["cases"][0]
    baseline, audit = prefix.read(BASELINE), prefix.read(BASELINE_AUDIT)
    prefix.require(
        baseline["complete"] is baseline["passed"] is baseline["sources_unchanged"] is True
        and baseline["phase"] == "producer"
        and len(baseline["branch_refs"]) == len(audit["cases"]) == 12
        and baseline["counts"]["totals"]["control_steps"] == 96
        and audit["complete"] is audit["passed"] is audit["sources_unchanged"] is True
        and audit["producer_ref"] == BASELINE
        and audit["original_commands_compared"] == audit["checker_commands_compared"] == 96
        and audit["checker_candidate_modules_unloaded"] == 12,
        "The independently replayed complete reset baseline is required",
    )
    branch_ref = baseline["branch_refs"][index]
    branch = prefix.read(branch_ref)
    row, case_id = audit["cases"][index], case["case"]
    periods = branch["runtime"]["continuation"]["periods"]
    prefix.require(
        row["case"] == case_id == branch["case"]
        and row["producer_ref"] == branch_ref
        and all(value == 0 for value in row["producer_vs_checker"].values())
        and branch["complete"] is branch["passed"] is True
        and case_id["case_index"] == index
        and case_id["case_id"] == current["case_id"]
        and case_id["sampling_seed"] == current["seed"]
        and case_id["variant"] == "raw"
        and case["record"]["task"] == TASKS[index // 4]
        and case["chunk"]["response_identity"]["seed"] == current["seed"]
        and len(periods) == 8
        and [item["step"] for item in periods] == list(range(8))
        and branch["runtime"]["continuation"]["stop"] is None,
        "The case, original reference seed and eight audited physical periods must match",
    )
    endpoint = periods[-1]["endpoint_receipt"]
    initial = dict(
        step=endpoint["endpoint_step"], sim_time_s=endpoint["sim_time"], state_sha256=endpoint["sim_state_sha256"]
    )
    prefix.require(initial["step"] == 8, "The model stream must start at the actual eighth endpoint")
    prepared, bridge, _old, template, _response = engine.previous_inputs()
    model_config = prepared["model_config"]
    source_root = Path(model_config["cosmos_source_root"]).resolve()
    model_sources = {
        path: digest for path, digest in bridge["source_sha256"].items() if Path(path).is_relative_to(source_root)
    }
    weights = {path: detail["sha256"] for path, detail in prepared["weight_files"].items()}
    fingerprint = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    checkpoint = model_config["checkpoint"]
    model_path = (Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model").resolve()
    prefix.require(
        source_root.is_relative_to(ROOT / "outputs")
        and source_root == Path(bridge["config"]["cosmos_source_root"]).resolve()
        and Path(model_config["model_python"]).is_file()
        and checkpoint["name"] == "8400_masked"
        and checkpoint["iteration"] == "iter_000008400"
        and Path(case["chunk"]["response_identity"]["checkpoint_path"]).resolve() == model_path
        and len(model_sources) == 1004
        and len(weights) == 163
        and fingerprint == config["model_sha256"],
        "The historical first reference and registered model/checkpoint manifests must agree",
    )
    refs = dict(
        config=config_ref(), plan=plan_ref(), baseline=BASELINE, baseline_audit=BASELINE_AUDIT, branch=branch_ref
    )
    consumed = dict(base["source_sha256"])
    for ref in refs.values():
        prefix.checked(ref, consumed)
    zero_physical = {name: dict(lower=0, upper=0) for name in ("control_steps", "native_steps", "models", "contexts")}
    zero_model = {
        name: dict(lower=0, upper=0)
        for name in ("initializations_completed", "requests_attempted", "requests_completed", "denoise_calls_completed")
    }
    return dict(
        config=current,
        base=base,
        case=case,
        baseline_periods=periods,
        initial=initial,
        refs=refs,
        source_sha256=consumed,
        prepared=prepared,
        bridge=bridge,
        template=template,
        model_source_hashes=model_sources,
        weight_hashes=weights,
        previous_physical_bounds=zero_physical,
        previous_model_bounds=zero_model,
    )


def input_receipt(inputs):
    return dict(
        refs=inputs["refs"],
        base=feedback.input_receipt(inputs["base"]),
        case=inputs["case"]["case"],
        initial=inputs["initial"],
        source_sha256=inputs["source_sha256"],
    )


def model_registration(inputs, image_root):
    config = inputs["config"]
    image_root = Path(image_root).resolve()
    case_root = ROOT / config["output"]
    allowed = {case_root / "producer/simulation", case_root / "checker/simulation"}
    prefix.require(image_root in allowed, "Each case must use its own producer or checker capture root")
    return dict(
        session=config["session"],
        initial=dict(inputs["initial"]),
        image_root=str(image_root.relative_to(ROOT)),
        task=inputs["case"]["record"]["task"],
        seed=config["seed"],
        model_sha256=config["model_sha256"],
        max_requests=config["max_model_requests"],
        max_periods=config["max_adaptive_periods"],
        denoise_steps=config["denoise_calls_per_request"],
    )


def make_trial(index):
    """Create one case from the common matrix without starting any process."""
    config = case_config(prefix.read(config_ref()), index)
    return registered.RegisteredAdaptiveTrial(
        output=ROOT / config["output"],
        qa=QA,
        dry=ROOT / config["dry"],
        module=MODULE,
        gpu_uuid=config["gpu_uuid"],
        required_qa_targets=QA_TARGETS,
        load_inputs=lambda: load_inputs(index),
        input_receipt=input_receipt,
        source_refs=source_refs,
        model_registration=model_registration,
    )
