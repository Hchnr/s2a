"""The physical diagnostic must bind the fixed correct-reference prefix."""

import numpy as np

from scripts.simulation.runtime import prepare_defer_close_correct as binding


def test_actual_input_binding_keeps_fixed_case_and_original_trigger():
    inputs = binding.load_inputs()
    assert inputs["spec"] == dict(case_index=8, case_id="correct_08", task_id=6, seed=42, prefix_steps=80)
    assert inputs["commands"].shape == (80, 7)
    assert inputs["fixed_continuation_commands"].shape == (8, 7)
    assert inputs["continuation_states"].shape == (8, 79)
    assert len(inputs["receipts"]) == len(inputs["saved"]["sim_states"]) == 81
    assert inputs["baseline_intervention_decision"]["command"] == 1.0
    assert inputs["fixed_continuation_commands"][5, 6] == 1.0
    assert inputs["targets"].shape == (40, 4, 4) and inputs["openings"].shape == (40,)
    assert all(np.isfinite(value).all() for value in (inputs["commands"], inputs["targets"], inputs["openings"]))
    assert not any(
        inputs["runtime_config"][key].startswith("/share/project/")
        for key in ("libero_root", "libero_config", "osmesa_library")
    )
