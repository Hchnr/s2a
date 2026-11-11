"""Freeze the three reset cases and bounded current-image WAM inputs."""

from pathlib import Path

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_second_admission as previous
from scripts.wam.tracking.preview.reset_multitask_live_requests import TASKS

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_multitask_live_48_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_multitask_live_48_20261111.md"
OUTPUT = ROOT / "outputs/reset_multitask_live_48_20261111"
QA = ROOT / "outputs/qa/reset_multitask_live_48_20261111/result.json"
DRY = ROOT / "outputs/reset_multitask_live_48_dry_20261111/result.json"
LIVE_AUDIT = {
    "path": str(ROOT / "outputs/reset_live_second_audit_20261111/result.json"),
    "sha256": "60e2d1b66403de596387798ea819ad94bc58e44b8b5656f99527e35e9532966c",
}
CASES = [
    dict(
        case_index=0,
        case_id="task0_init0_s42_raw",
        task=TASKS[0],
        target="akita_black_bowl_1",
        gpu_uuid="GPU-45f7c2db-5c3a-91f8-a299-e21c0681a71b",
    ),
    dict(
        case_index=4,
        case_id="task1_init0_s42_raw",
        task=TASKS[1],
        target="wine_bottle_1",
        gpu_uuid="GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf",
    ),
    dict(
        case_index=8,
        case_id="task2_init0_s42_raw",
        task=TASKS[2],
        target="cream_cheese_1",
        gpu_uuid="GPU-5f5f2b7d-a87b-28e2-0bcc-ee85f2889db4",
    ),
]
FIXED = dict(
    schema="s2a.reset-multitask-live-48.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    cases=CASES,
    max_parallel_cases=3,
    seed=42,
    first_window_steps=8,
    live_window_steps=8,
    max_model_initializations_per_case=1,
    max_model_requests_per_case=5,
    denoise_steps=35,
    max_control_steps_per_case_per_phase=48,
    max_native_steps_per_case_per_phase=1200,
    max_models_per_case_per_phase=3,
    max_contexts_per_case_per_phase=3,
    max_quality_responses_per_period=320,
    max_baseline_responses_per_period=480,
    max_gripper_sequences_per_period=6561,
    max_audit_endpoint_queries_per_case=9600,
    max_audit_transition_steps_per_case=124800,
    model_sha256=previous.FIXED["model_sha256"],
    worker_timeout_seconds=3600,
    fresh_cases=0,
    fits=0,
    automatic_retry=False,
)


def _exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_exact(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_exact(a, b) for a, b in zip(actual, expected, strict=True))
    return actual == expected


def validate_config(config):
    prefix.require(_exact(config, FIXED), "The exact three-case roster, GPU bindings and budgets are required")


def source_refs():
    names = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/run_reset_multitask_live_48.py",
        ROOT / "scripts/simulation/runtime/reset_live_multi_window.py",
        ROOT / "scripts/simulation/runtime/reset_multitask_live_audit.py",
        ROOT / "scripts/simulation/runtime/reset_multitask_live_workers.py",
        ROOT / "scripts/simulation/runtime/reset_live_multi_window_replay.py",
        ROOT / "scripts/simulation/runtime/reset_live_two_window_replay_v2.py",
        ROOT / "scripts/wam/tracking/preview/reset_multitask_live_requests.py",
        ROOT / "src/s2a/research/native_runtime/reset_multi_window_runtime.py",
        ROOT / "src/s2a/research/native_runtime/development_prefix_runtime_48.py",
        ROOT / "tests/research/native_runtime/test_reset_multitask_live_admission.py",
        ROOT / "tests/research/native_runtime/test_reset_multitask_live_owner.py",
        ROOT / "tests/research/native_runtime/test_reset_multitask_live_audit.py",
        ROOT / "tests/research/native_runtime/test_reset_multitask_live_workers.py",
        ROOT / "tests/research/native_runtime/test_reset_live_multi_window.py",
        ROOT / "tests/research/native_runtime/test_reset_live_multi_window_replay.py",
        ROOT / "tests/research/native_runtime/test_reset_multi_window_runtime.py",
        ROOT / "tests/research/wam/test_reset_multitask_live_requests.py",
    )
    return previous.source_refs() + [prefix.reference(path) for path in names]


def validate_case(spec, case, branch):
    identity = case["case"]
    periods = branch["runtime"]["continuation"]["periods"]
    prefix.require(
        identity["case_index"] == spec["case_index"]
        and identity["case_id"] == spec["case_id"]
        and identity["sampling_seed"] == 42
        and identity["variant"] == "raw"
        and identity["task_index"] == spec["case_index"] // 4
        and case["record"]["task"] == spec["task"]
        and case["record"]["task_index"] == identity["task_index"]
        and case["record"]["initial_index"] == 0
        and case["target_object"] == spec["target"]
        and branch["complete"] is branch["passed"] is True
        and branch["case"] == identity
        and len(periods) == 8
        and [period["step"] for period in periods] == list(range(8))
        and branch["runtime"]["continuation"]["stop"] is None,
        "Each registered task must use its matching completed raw reset baseline",
    )
    end = periods[-1]["endpoint_receipt"]
    initial = {"step": end["endpoint_step"], "sim_time_s": end["sim_time"], "state_sha256": end["sim_state_sha256"]}
    prefix.require(initial["step"] == 8, "The current-image model starts after eight actual periods")
    return initial


def load_inputs(*, verify_assets=True):
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    prior = previous.load_inputs(verify_assets=verify_assets)
    baseline, audit = prefix.read(previous.BASELINE), prefix.read(LIVE_AUDIT)
    prefix.require(
        audit["complete"] is audit["passed"] is audit["sources_unchanged"] is True
        and audit["original_failure_preserved"] is True
        and audit["producer_control_steps"] == audit["repaired_checker_control_steps"] == 16
        and audit["model_initializations"] == audit["model_requests"] == 1
        and audit["denoise_calls"] == 35
        and audit["fresh_cases"] == 0,
        "The previous actual current-image pilot and independent replay must be complete",
    )
    selected = {}
    consumed = dict(prior["source_sha256"])
    for ref in (config_ref, LIVE_AUDIT):
        prefix.checked(ref, consumed)
    for spec in config["cases"]:
        index = spec["case_index"]
        case = prior["base"]["cases"][index]
        branch_ref = baseline["branch_refs"][index]
        branch = prefix.read(branch_ref, consumed)
        initial = validate_case(spec, case, branch)
        selected[index] = dict(
            case=case,
            branch_ref=branch_ref,
            baseline_periods=branch["runtime"]["continuation"]["periods"],
            initial=initial,
        )
    return dict(
        previous=prior,
        config=config,
        config_ref=config_ref,
        selected=selected,
        source_sha256=consumed,
        prepared=prior["prepared"],
        bridge=prior["bridge"],
        template=prior["template"],
        model_source_hashes=prior["model_source_hashes"],
        weight_hashes=prior["weight_hashes"],
    )


def input_receipt(inputs):
    return dict(
        previous=previous.input_receipt(inputs["previous"]),
        config_ref=inputs["config_ref"],
        prior_live_audit=LIVE_AUDIT,
        cases=[
            dict(
                spec=spec,
                case=inputs["selected"][spec["case_index"]]["case"]["case"],
                baseline_branch=inputs["selected"][spec["case_index"]]["branch_ref"],
                initial=inputs["selected"][spec["case_index"]]["initial"],
            )
            for spec in inputs["config"]["cases"]
        ],
        source_sha256=inputs["source_sha256"],
    )


def model_registration(inputs, index, image_root):
    spec = next((item for item in inputs["config"]["cases"] if item["case_index"] == index), None)
    prefix.require(spec is not None, "Only registered task cases are admitted")
    return dict(
        session=f"reset_multitask_case{index}_live_s42_20261111",
        initial=inputs["selected"][index]["initial"],
        image_root=str(Path(image_root).relative_to(ROOT)),
        task=spec["task"],
        seed=inputs["config"]["seed"],
        model_sha256=inputs["config"]["model_sha256"],
        max_requests=inputs["config"]["max_model_requests_per_case"],
        denoise_steps=inputs["config"]["denoise_steps"],
    )
