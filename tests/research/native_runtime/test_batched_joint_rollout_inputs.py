"""The benchmark must reuse exactly the prior causal profiling inputs."""

from scripts.simulation.runtime import prepare_batched_joint_rollout as binding


def test_real_inputs_match_the_three_previously_profiled_samples():
    inputs = binding.load_inputs()
    receipt = binding.input_receipt(inputs)
    assert inputs["case"]["case_id"] == "task1_init0_s42_terminal058"
    assert list(receipt["sample_fingerprints"]) == ["352", "355", "359"]
    assert all(len(value) == 64 for value in receipt["sample_fingerprints"].values())
    assert inputs["config"]["enforce_limits"] is False and inputs["config"]["solver_calls"] == 0
