"""Guard the causal boundary of the emergency comparison inputs."""

import sys

import numpy as np

from scripts.simulation.runtime import compare_multicase_emergency_retreat as compare


def test_reference_decoder_does_not_parse_later_queries(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text('[{"step": 0}, {"step": 8}, invalid-future-data]')
    assert compare.read_array_item(path, 1) == {"step": 8}


def test_checker_replays_recorded_action_without_importing_policy(monkeypatch):
    monkeypatch.delitem(sys.modules, compare.POLICY_MODULE, raising=False)
    command = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0]
    action, historical_states = compare.make_action("retreat", {}, expected_periods=[{"command": command}])
    result = action(None, None, None, None, 0)
    np.testing.assert_array_equal(result["command"], command)
    assert result["newly_selected"] is False
    assert historical_states is None
    assert compare.POLICY_MODULE not in sys.modules


def test_zero_candidate_does_not_read_historical_future(monkeypatch):
    def reject_future(_case):
        raise AssertionError("Candidate must not read historical future")

    monkeypatch.setattr(compare, "history_inputs", reject_future)
    contract = {
        "input_min": [-1.0] * 6,
        "input_max": [1.0] * 6,
        "output_min": [-0.05] * 6,
        "output_max": [0.05] * 6,
    }
    action, historical_states = compare.make_action("zero", {"controller": {"contract": contract}})
    result = action(np.zeros(7), np.tile([-1.0, 1.0], (7, 1)), np.zeros((6, 7)), np.zeros(7), 0)
    np.testing.assert_array_equal(result["command"], np.zeros(7))
    assert result["newly_selected"] is True
    assert historical_states is None
