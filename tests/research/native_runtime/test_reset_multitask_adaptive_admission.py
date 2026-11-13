"""Admit the fixed six-case adaptive development matrix without physics."""

import copy

import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_multitask_adaptive_admission as admission


@pytest.mark.parametrize("index", [0, 2, 4, 6, 8, 10])
def test_each_real_reset_and_seed_bind_the_common_full_run(index):
    inputs = admission.load_inputs(index, verify_assets=False)
    trial = admission.make_trial(index)
    trial.require_config(inputs)
    registration = admission.model_registration(inputs, trial.output / "producer/simulation")
    assert inputs["case"]["case"]["case_index"] == index
    assert inputs["case"]["case"]["sampling_seed"] == registration["seed"] == (42 if index % 4 == 0 else 43)
    assert inputs["case"]["record"]["task"] == registration["task"]
    assert inputs["initial"]["step"] == registration["initial"]["step"] == 8
    assert len(inputs["baseline_periods"]) == 8
    assert registration["max_periods"] == 592 and registration["max_requests"] == 75
    assert registration["initial"]["step"] + registration["max_periods"] == 600
    assert len(inputs["model_source_hashes"]) == 1004 and len(inputs["weight_hashes"]) == 163
    assert inputs["previous_physical_bounds"]["control_steps"] == {"lower": 0, "upper": 0}
    assert inputs["previous_model_bounds"]["requests_completed"] == {"lower": 0, "upper": 0}
    assert trial.model_bundle(inputs)["model_registration"] == registration


@pytest.mark.parametrize("change", ["case", "seed", "gpu", "budget", "boolean"])
def test_config_changes_do_not_change_the_registered_experiment(change):
    config = copy.deepcopy(prefix.read(admission.config_ref()))
    if change == "case":
        config["cases"][0]["case_id"] = "other"
    elif change == "seed":
        config["cases"][0]["sampling_seed"] = 43
    elif change == "gpu":
        config["cases"][0]["gpu_uuid"] = config["cases"][1]["gpu_uuid"]
    elif change == "budget":
        config["max_adaptive_periods"] = 40
    else:
        config["fresh_cases"] = False
    with pytest.raises(ValueError):
        admission.validate_config(config)


@pytest.mark.parametrize("index", [-1, 1, True, 12, "0"])
def test_unregistered_cases_are_rejected(index):
    with pytest.raises(ValueError, match="registered case"):
        admission.make_trial(index)


def test_current_sources_include_the_shared_controller_and_model_contract():
    paths = {item["path"] for item in admission.source_refs()}
    assert str(admission.ROOT / "scripts/simulation/runtime/reset_adaptive_stream.py") in paths
    assert str(admission.ROOT / "scripts/wam/tracking/preview/adaptive_multiseed_model_worker.py") in paths
    assert all(str(admission.ROOT / name) in paths for name in admission.QA_TARGETS)
