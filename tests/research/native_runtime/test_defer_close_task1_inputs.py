"""Bind the actual previously checked task1 release case."""

from scripts.simulation.runtime.prepare_defer_close_task1 import load_inputs


def test_original_baseline_and_reference_are_bound_to_one_fixed_case():
    inputs = load_inputs()
    assert inputs["case"]["spec"]["case_id"] == "task1_init0_s43_terminal058"
    assert inputs["case"]["commands"].shape == (232, 7)
    assert inputs["selected"]["targets"].shape == (40, 4, 4)
    assert inputs["selected"]["cursor"] == 0
    assert inputs["baseline"]["runtime"]["continuation"]["periods"][2]["command"][6] == 1.0
