import copy
import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime import run_causal_wam_two_window_feedback as study


def test_two_references_bind_audited_actual_feedback_endpoints():
    inputs = study.load_inputs()
    assert [image["frame_index"] for image in inputs["images"]] == [80, 88]
    assert [target.shape for target in inputs["targets"]] == [(40, 4, 4), (40, 4, 4)]
    assert [opening.shape for opening in inputs["openings"]] == [(40,), (40,)]
    np.testing.assert_allclose(inputs["targets"][0], inputs["old"]["targets"], atol=1e-12, rtol=0)
    assert (
        inputs["images"][1]["current_endpoint"]["sim_state_sha256"]
        == inputs["periods"][-1]["endpoint_receipt"]["sim_state_sha256"]
    )


def test_first_action_must_match_before_dispatch():
    expected = {"command": np.arange(7, dtype=float) / 10}
    study.require_previous_action({"command": expected["command"].copy(), "newly_selected": True}, expected)
    with pytest.raises(ValueError, match="previously selected action"):
        study.require_previous_action({"stop": {"reason": "no_witness"}}, expected)
    wrong = expected["command"].copy()
    wrong[0] += 0.01
    with pytest.raises(AssertionError):
        study.require_previous_action({"command": wrong, "newly_selected": True}, expected)


def test_first_period_rejects_changed_actual_inner_sample():
    period = study.load_inputs()["periods"][0]
    study.compare_first_period(copy.deepcopy(period), period)
    changed = copy.deepcopy(period)
    changed["actual_inner"][0]["q"][0] += 0.01
    with pytest.raises(AssertionError):
        study.compare_first_period(changed, period)


def test_both_render_callbacks_pass_observer_and_reject_changed_image(tmp_path):
    images = study.load_inputs()["images"]
    runtime = tmp_path / "runtime"
    runtime.mkdir()

    class Observer:
        @staticmethod
        def render_anchor(env, step, sample, endpoint, destination):
            assert env == "live" and sample == "sample" and endpoint == "endpoint"
            (destination / f"anchor{step:04d}").mkdir()
            return copy.deepcopy(images[0 if step == 80 else 1])

    for image in images:
        result = study.render_and_verify(Observer, "live", image["frame_index"], "sample", "endpoint", runtime, image)
        assert result["frame_index"] == image["frame_index"]
    wrong = copy.deepcopy(images[1])
    wrong["canvas_rgb_sha256"] = "other"
    with pytest.raises(ValueError, match="corresponding WAM request"):
        study.validate_live_capture(images[1], wrong)


def test_checker_does_not_import_candidate_policies():
    source = (
        "from scripts.simulation.runtime import run_causal_wam_two_window_feedback as s; "
        "s.load_inputs(); s.prior.checker_isolation(); print('isolated')"
    )
    child = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, check=True, timeout=90)
    assert "isolated" in child.stdout
