"""The same-predictor baseline must retain the complete original experiment."""

import copy
import json
from pathlib import Path

import pytest

from scripts.tracking.trajectory_search.check_replay_order_sumq import validate_spec


@pytest.fixture
def configurations():
    root = Path(__file__).resolve().parents[3]
    folder = root / "configs/tracking/trajectory_search"
    original = json.loads((folder / "state_vector_dual_recovery_20261029.json").read_bytes())
    spec = json.loads((folder / "replay_order_sumq_20261102.json").read_bytes())
    return spec, original


def test_original_configuration_is_preserved(configurations):
    spec, original = configurations
    assert validate_spec(spec, original) == original


@pytest.mark.parametrize("change", ["minimum_reduction", "response_budget", "retry", "tool_policy"])
def test_gate_or_budget_changes_are_rejected(configurations, change):
    spec, original = copy.deepcopy(configurations)
    if change == "minimum_reduction":
        spec["search_config"]["minimum_actual_reduction"] *= 0.1
    elif change == "response_budget":
        spec["total_response_budget"] += 1
    elif change == "retry":
        spec["automatic_retry"] = True
    else:
        spec["tool_policy"] = "historical_transpose"
    with pytest.raises(ValueError):
        validate_spec(spec, original)
