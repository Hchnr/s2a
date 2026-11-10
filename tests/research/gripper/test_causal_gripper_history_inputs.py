"""Bind reconstructed controller history to the verified saved prefixes."""

import pytest

from scripts.simulation.runtime import reconstruct_causal_gripper_history as history


def test_actual_inputs_bind_both_verified_seed42_prefixes():
    inputs = history.load_inputs()
    assert [item["spec"]["case_index"] for item in inputs["cases"]] == [0, 5]
    assert [item["spec"]["prefix_steps"] for item in inputs["cases"]] == [136, 360]
    assert all("/development/cases/" in item["steps_ref"]["path"] for item in inputs["cases"])
    assert all("/final/" not in path for path in inputs["source_sha256"])


def test_chunk_reader_does_not_decode_a_future_item(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text('[{"step":0}, {"step":8}, invalid future]')
    assert history.read_chunk_prefix(path, 2) == [{"step": 0}, {"step": 8}]
    with pytest.raises(ValueError):
        history.read_chunk_prefix(path, 3)
    for count in (0, True, 1.5):
        with pytest.raises(ValueError):
            history.read_chunk_prefix(path, count)


def test_registered_costs_cover_exact_historical_prefixes():
    _, config = history.read_config()
    assert sum(item["prefix_steps"] for item in config["cases"]) == config["max_reconstructed_steps"] == 496
    assert config["max_candidate_sequences"] == config["max_planner_calls"] * 3**8
    assert config["physics_steps"] == config["world_model_calls"] == config["fresh_cases"] == 0
