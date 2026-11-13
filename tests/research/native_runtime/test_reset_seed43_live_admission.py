"""Check seed-43 reset, historical evidence and step-8 model admission."""

import copy

import pytest

from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalModelAdmission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_seed43_live_admission as admission


@pytest.fixture(scope="module")
def saved():
    inputs = admission.load_inputs(verify_assets=True)
    config, refs = inputs["config"], inputs["refs"]
    baseline = prefix.read(refs["historical_segmented_producer"])
    audit = prefix.read(refs["historical_segmented_audit"])
    branch = prefix.read(refs["historical_case_branch"])
    equivalence = prefix.read(refs["reference_equivalence_audit"])
    return inputs, config, baseline, audit, branch, equivalence


def test_recorded_case_and_model_registration_load_without_physical_execution(saved):
    inputs, config, _baseline, _audit, _branch, _equivalence = saved
    assert inputs["case"]["case"]["case_id"] == "task0_init0_s43_raw"
    assert inputs["initial"] == {
        "step": 8,
        "sim_time_s": 0.6500000000000005,
        "state_sha256": config["historical_step8_state_sha256"],
    }
    assert len(inputs["baseline_periods"]) == 8 and len(inputs["model_source_hashes"]) == 1004
    assert len(inputs["weight_hashes"]) == 163
    registration = admission.model_registration(inputs, admission.OUTPUT / "producer/simulation/runtime")
    service = CausalModelAdmission(**registration)
    assert service.phase == "ready" and service.seed == 43 and service.max_requests == 1
    assert registration["task"] == "put the bowl on the plate" and registration["denoise_steps"] == 35


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("case_index", 8),
        ("sampling_seed", 42),
        ("sampling_seed", True),
        ("task", "put the cream cheese in the bowl"),
        ("gpu_uuid", "GPU-unregistered"),
        ("max_model_requests", 2),
        ("max_native_steps_per_phase", 401),
        ("fresh_cases", False),
        ("automatic_retry", True),
    ],
)
def test_registered_case_seed_and_budgets_cannot_be_changed(key, value):
    config = prefix.read({"path": str(admission.CONFIG), "sha256": admission.CONFIG_SHA})
    config[key] = value
    with pytest.raises(ValueError, match="exact registered"):
        admission.validate_config(config)


@pytest.mark.parametrize("change", ["case", "reset", "reference", "endpoint", "audit", "equivalence"])
def test_corrupted_historical_binding_is_rejected(saved, change):
    inputs, config, original_baseline, original_audit, original_branch, original_equivalence = saved
    case = copy.deepcopy(inputs["case"])
    base = {"cases": [None, None, case]}
    baseline, audit, branch, equivalence = map(
        copy.deepcopy, (original_baseline, original_audit, original_branch, original_equivalence)
    )
    if change == "case":
        case["case"]["sampling_seed"] = 42
    elif change == "reset":
        case["expected_receipt"]["sim_state_sha256"] = "0" * 64
    elif change == "reference":
        case["chunk"]["response_sha256"] = "0" * 64
    elif change == "endpoint":
        branch["runtime"]["continuation"]["periods"][-1]["endpoint_receipt"]["sim_state_sha256"] = "0" * 64
    elif change == "audit":
        audit["checker_commands_compared"] = 95
    else:
        equivalence["historical_short_window"]["replay_max_absolute_difference"]["command"] = 0.01
    with pytest.raises(ValueError):
        admission.check_history(config, base, baseline, audit, branch, equivalence)


def test_wrong_current_image_root_cannot_be_registered(saved, tmp_path):
    with pytest.raises(ValueError, match="current-image root"):
        admission.model_registration(saved[0], tmp_path)
