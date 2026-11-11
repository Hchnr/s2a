import copy
import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime import run_saved_wam_feedback as study


def test_saved_model_reference_replaces_correct_targets_at_the_same_anchor():
    inputs = study.load_inputs()
    assert inputs["targets"].shape == (40, 4, 4)
    assert inputs["openings"].shape == (40,)
    np.testing.assert_allclose(inputs["anchor"], inputs["base"]["first_anchor"], atol=1e-10, rtol=0)
    online = study.model_policy_inputs(inputs, inputs["anchor"])
    np.testing.assert_array_equal(online["targets"][0], inputs["targets"])
    np.testing.assert_array_equal(online["openings"][0], inputs["openings"])
    assert not np.array_equal(online["targets"][0], inputs["base"]["targets"][0])


def test_live_capture_rejects_different_state_or_camera_pixels():
    source = study.load_inputs()["source_capture"]
    study.validate_live_capture(source, copy.deepcopy(source))
    for key in ("sim_state_sha256", "canvas_rgb_sha256"):
        changed = copy.deepcopy(source)
        if key == "sim_state_sha256":
            changed["current_endpoint"][key] = "other"
        else:
            changed[key] = "other"
        with pytest.raises(ValueError, match="exactly match"):
            study.validate_live_capture(source, changed)


def test_empty_predispatch_stop_is_bounded_but_extra_native_steps_are_rejected():
    config = study.read_config()[1]
    runtime = {
        "control_steps_completed": 80,
        "counters": {"mj_step": {"completed": 2000}},
        "continuation": {"periods": []},
        "newly_selected_commands": 0,
        "prefix_inner_periods": [{"samples": [None] * 25} for _ in range(80)],
        "checked_endpoints": 81,
        "cleanup_complete": True,
    }
    branch = {"complete": True, "passed": True, "runtime": runtime, "candidate_modules_unloaded": True}
    assert study.summarize_phase([branch], "checker", config)["continuation_steps"] == 0
    runtime["counters"]["mj_step"]["completed"] = 2025
    with pytest.raises(ValueError, match="Bounded actual feedback"):
        study.summarize_phase([branch], "checker", config)


def test_checker_does_not_import_candidate_policies():
    source = (
        "from scripts.simulation.runtime import run_saved_wam_feedback as s; "
        "s.load_inputs(); s.prior.checker_isolation(); print('isolated')"
    )
    child = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, check=True, timeout=90)
    assert "isolated" in child.stdout
