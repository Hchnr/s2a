"""Verify the new seed-43 sample's link to prior physical feedback evidence."""

import copy

import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.wam.model_contracts import audit_seed43_historical_control_20261113 as audit


@pytest.fixture(scope="module")
def saved():
    new, old, history = (prefix.read(ref) for ref in (audit.NEW, audit.V4, audit.HISTORICAL_AUDIT))
    model = prefix.read(new["model_ref"])
    old_request, old_response = prefix.read(old["request_ref"]), prefix.read(old["response_ref"])
    new_request = prefix.read(model["responses"][0]["model_request_ref"])
    new_response = prefix.read(model["responses"][0]["actual_response_ref"])
    case = history["cases"][2]
    produced, replayed = prefix.read(case["producer_ref"]), prefix.read(case["checker_ref"])
    return old_request, old_response, new_request, new_response, produced, replayed


def test_historical_reference_and_short_window_are_bound_without_new_execution():
    result = audit.audit()
    assert result["complete"] and result["passed"] and not result["control_qualified"]
    assert result["evidence_level"] == "retrospective_reference_equivalence_not_new_closed_loop"
    assert result["new_model_requests"] == result["new_simulation_models"] == result["new_physics_steps"] == 0
    for comparison in result["same_saved_reference"].values():
        assert all(row["elementwise_equal"] and row["different_elements"] == 0 for row in comparison.values())
    metrics = result["historical_short_window"]
    assert metrics["control_periods"] == metrics["pose_cap_endpoints"] == 8 and metrics["inner_samples"] == 200
    assert metrics["task_success_endpoints"] == metrics["target_grasp_endpoints"] == 0
    assert metrics["position_rmse_m"] == pytest.approx(0.0013911670565759315)
    assert metrics["rotation_rmse_rad"] == pytest.approx(0.012129853632720573)
    assert all(value == 0 for value in metrics["replay_max_absolute_difference"].values())


@pytest.mark.parametrize("change", ["seed", "task", "sampling", "array"])
def test_changed_reference_cannot_inherit_historical_feedback(saved, change):
    old_request, old_response, new_request, new_response, _produced, _replayed = saved
    request, response = copy.deepcopy(new_request), copy.deepcopy(new_response)
    if change == "seed":
        request["seed"] = 42
    elif change == "task":
        request["prompt"] = "different task"
    elif change == "sampling":
        request["num_steps"] = 34
    else:
        response["outputs"][0]["content"]["action"][0][0] += 0.01
    with pytest.raises(ValueError):
        audit.same_request_and_response(old_request, old_response, request, response)


def test_independent_replay_comparison_detects_changed_command_and_endpoint(saved):
    _a, _b, _c, _d, produced, replayed = saved
    original = produced["runtime"]["continuation"]["periods"]
    changed = copy.deepcopy(replayed["runtime"]["continuation"]["periods"])
    changed[3]["command"][0] += 0.01
    result = audit.compare_replays(original, changed)
    assert result["replay_max_absolute_difference"]["command"] == pytest.approx(0.01)
    changed[3]["endpoint_receipt"]["sim_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="endpoints"):
        audit.compare_replays(original, changed)
