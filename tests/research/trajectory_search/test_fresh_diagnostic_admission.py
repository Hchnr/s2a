"""New replay admission cannot replace old failures or change the experiment."""

import copy
import json

import pytest

from scripts.tracking.trajectory_search.state_vector_fresh_diagnostic_admission import CONFIG, validate_spec


def test_registered_replay_preserves_original_failed_and_producer_identities():
    spec = json.loads(CONFIG.read_text())
    before = copy.deepcopy(spec)
    result = validate_spec(spec)
    assert result == spec == before and result is not spec
    assert result["expected_queries"] == 1 + 312 * result["expected_responses"]
    assert result["expected_inner_steps"] == 300 * result["expected_responses"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("numeric_tolerance", 1e-9),
        ("expected_responses", 121),
        ("automatic_retry", 0),
        ("tool_transform_policy", "saved_producer_input"),
        ("output_dir", "outputs/tracking_v5_state_vector_dual_recovery_check_20261029"),
    ],
)
def test_diagnostic_replay_cannot_relax_gates_change_inputs_or_overwrite_failure(key, value):
    spec = json.loads(CONFIG.read_text())
    spec[key] = value
    with pytest.raises(ValueError, match="Fixed diagnostic contract"):
        validate_spec(spec)


def test_replay_cannot_relabel_failed_checker_as_a_different_artifact():
    spec = json.loads(CONFIG.read_text())
    spec["failed_checker_ref"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Original failed/producer identity"):
        validate_spec(spec)
