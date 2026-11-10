"""Solver comparison reuses the full-output-checked causal rollout inputs."""

from scripts.simulation.runtime import prepare_batched_quality_solver as binding


def test_real_samples_bind_completed_rollout_benchmarks():
    inputs = binding.load_inputs()
    assert inputs["case"]["case_id"] == "task1_init0_s42_terminal058"
    assert list(inputs["samples"]) == ["352", "355", "359"]
    assert inputs["config"]["past_inner_error"] == "zero_diagnostic_only"
    assert all(len(value) == 64 for value in binding.input_receipt(inputs)["sample_fingerprints"].values())
