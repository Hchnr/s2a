"""The hold pilot must use the previously replayed case and frozen inputs."""

import numpy as np

from scripts.simulation.runtime import prepare_bounded_release_hold as binding
from scripts.simulation.runtime import prepare_causal_event_bridge_feedback as previous


def test_physical_pilot_binds_the_exact_replayed_state_and_reference_times():
    inputs = binding.load_inputs()
    config, case = inputs["config"], inputs["case"]
    assert config["case_id"] == case["spec"]["case_id"] == "task1_init0_s42_terminal058"
    assert len(case["commands"]) == config["prefix_steps"] == 360
    old, latest = previous.selected_references(case)
    assert old["cursor"] == 16 and latest["cursor"] == 0
    assert old["targets"].shape == latest["targets"].shape == (40, 4, 4)
    assert np.isfinite(old["targets"]).all() and np.isfinite(latest["targets"]).all()
    assert set(inputs["prior_long_refs"]) == set(inputs["prior_shadow_refs"]) == {"producer", "checker"}
