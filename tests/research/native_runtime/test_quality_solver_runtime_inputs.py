"""Profile inputs must remain causal and preserve measured state and commands."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_quality_solver_runtime_profile as profile


def test_real_samples_bind_current_state_and_existing_generated_reference():
    inputs = profile.load_inputs()
    assert list(inputs["samples"]) == ["352", "355", "359"]
    for step, sample in inputs["samples"].items():
        assert sample["step"] == int(step) and sample["reference_step"] == 352
        assert sample["reference_offset"] == int(step) - 352
        assert sample["targets"].shape == (8, 4, 4)
        assert sample["seed"].shape == (8, 6)
        assert not np.any(sample["past_inner_error_rad"])
        assert len(profile.sample_fingerprint(sample)) == 64
    changed = copy.deepcopy(inputs["samples"]["352"])
    changed["q"][0] += 0.001
    assert profile.sample_fingerprint(changed) != profile.sample_fingerprint(inputs["samples"]["352"])


def test_rejects_reference_or_state_misalignment():
    inputs = profile.binding.load_inputs()
    case = inputs["case"]
    config = profile.read_config()[1]
    chunk = profile.saved_chunk.read_array_item(
        profile.prefix.checked(case["gripper_inputs"]["chunks_ref"]), config["reference_step"] // 8
    )
    result = profile.prefix.read(profile.RECOVERY_RESULTS["producer"])
    runtime = profile.prefix.read(result["branch_refs"][profile.binding.MODES[0]])["runtime"]
    row, previous = copy.deepcopy(case["rows"][352]), case["rows"][351]
    row["qualified_forward_input"]["x_joint_position"][0][0] += 0.01
    with pytest.raises(AssertionError):
        profile.prepare_sample(case, row, previous, chunk, runtime, 352, config)
    row = case["rows"][352]
    chunk["step"] = 360
    with pytest.raises(ValueError):
        profile.prepare_sample(case, row, previous, chunk, runtime, 352, config)
