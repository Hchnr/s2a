"""A physical replay must bind only the registered development prefixes."""

import copy

import pytest

from scripts.simulation.runtime import replay_causal_event_prefixes as replay


def synthetic_opportunity():
    spec = replay.CASES[0]
    shadow_records = [None] * (spec["prefix_steps"] // 8 + 1)
    shadow_records[-1] = {
        "step": spec["prefix_steps"],
        "source": "remembered",
        "source_step": spec["source_step"],
        "event": {"direction": "closure", "absolute_endpoint": spec["event_endpoint"]},
        "baseline_observation_label": "first_intervention_on_baseline",
    }
    case = {"case_index": spec["case_index"], "case_id": spec["case_id"]}
    shadow = {"case": case, "first_intervention_step": spec["prefix_steps"], "records": shadow_records}
    independent = {"case": case, "first_intervention_step": spec["prefix_steps"]}
    geometry_queries = [None] * (spec["source_step"] // 8 + 1)
    geometry_queries[-1] = {
        "step": spec["source_step"],
        "first_closure_index": spec["event_endpoint"] - spec["source_step"] - 1,
        "variants": [{"name": "predicted", "pad_contacts": [True, True]}, {"name": "zero"}],
    }
    return spec, shadow, independent, {"case": case, "queries": geometry_queries}


def test_registered_seed42_inputs_exclude_future_states_and_commands():
    inputs = replay.load_inputs()
    assert [c["spec"]["case_index"] for c in inputs["cases"]] == [0, 5]
    assert len(inputs["asset_refs"]) == 96
    for case, expected in zip(inputs["cases"], replay.CASES, strict=True):
        n = expected["prefix_steps"]
        assert case["config"]["seed"] == 42 and case["config"]["task_id"] == expected["task_id"]
        assert case["commands"].shape == (n, 7)
        assert case["saved"]["sim_states"].shape == (n + 1, 79)
        assert len(case["receipts"]) == len(case["saved"]["gripper_current_action"]) == n + 1
    assert all("/final/" not in path for path in inputs["source_sha256"])


@pytest.mark.parametrize("corruption", ["case", "first", "source", "direction", "endpoint", "label", "geometry"])
def test_rejects_changed_source_or_unchecked_first_opportunity(corruption):
    spec, shadow, independent, geometry = copy.deepcopy(synthetic_opportunity())
    record = shadow["records"][-1]
    if corruption == "case":
        independent["case"]["case_id"] = "wrong"
    elif corruption == "first":
        independent["first_intervention_step"] += 8
    elif corruption == "source":
        record["source_step"] += 8
    elif corruption == "direction":
        record["event"]["direction"] = "opening"
    elif corruption == "endpoint":
        record["event"]["absolute_endpoint"] += 1
    elif corruption == "label":
        record["baseline_observation_label"] = "baseline_observation_after_intervention"
    else:
        geometry["queries"][-1]["first_closure_index"] += 1
    with pytest.raises(ValueError):
        replay.validate_shadow_case(spec, shadow, independent, geometry)


def test_both_saved_prefixes_fit_the_registered_physics_budget():
    _, config = replay.read_config()
    assert sum(case["prefix_steps"] for case in config["cases"]) == config["max_control_steps"] == 496
    assert config["max_native_steps"] == 25 * config["max_control_steps"]
    assert config["world_model_calls"] == config["fresh_cases"] == 0
