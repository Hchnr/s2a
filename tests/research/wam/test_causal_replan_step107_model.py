import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import run_causal_replan_step107_model as owner


def valid_capture():
    snapshot = dict(step=107, sim_time_s=5.6, state_sha256=owner.STATE_SHA)
    binding = {"sim_time": snapshot["sim_time_s"], "sim_state_sha256": owner.STATE_SHA}
    return dict(
        complete=True,
        passed=True,
        sources_unchanged=True,
        control_qualified=False,
        counts=dict(
            control_steps=107,
            native_steps=2675,
            stop=dict(reason="requested_observation", at_step=107, next_action_dispatched=False),
        ),
        independent_max_absolute_difference={"command": 0.0, "state": 0.0},
        terminal_capture=dict(
            snapshot=snapshot,
            capture=dict(
                frame_index=107,
                current_endpoint={"endpoint_step": 107, **binding},
                render_before=dict(binding),
                render_after=dict(binding),
                future_rgb_frames=0,
                canvas_png={"path": "unused.png", "sha256": owner.IMAGE_SHA},
                camera_roles=["auxiliary_0", "left_wrist"],
                canvas_shape=[960, 640, 3],
            ),
        ),
    )


def test_accepts_only_the_replayed_current_step107_capture():
    snapshot, capture = owner.validate_capture(valid_capture())
    assert snapshot["step"] == 107 and capture["canvas_png"]["sha256"] == owner.IMAGE_SHA


@pytest.mark.parametrize("mutation", ["step", "state", "image", "future", "stop", "difference", "cameras"])
def test_rejects_mismatched_model_input_before_any_request(mutation):
    result = valid_capture()
    capture = result["terminal_capture"]["capture"]
    if mutation == "step":
        result["terminal_capture"]["snapshot"]["step"] = 108
    elif mutation == "state":
        capture["render_after"]["sim_state_sha256"] = "a" * 64
    elif mutation == "image":
        capture["canvas_png"]["sha256"] = "a" * 64
    elif mutation == "future":
        capture["future_rgb_frames"] = 1
    elif mutation == "stop":
        result["counts"]["stop"]["reason"] = "quality_no_witness"
    elif mutation == "difference":
        result["independent_max_absolute_difference"]["state"] = 0.01
    else:
        capture["camera_roles"] = ["left_wrist", "auxiliary_0"]
    with pytest.raises(ValueError, match="step-107 current image and state"):
        owner.validate_capture(result)


def test_binding_preserves_the_original_model_owner_and_device():
    code = (
        "from scripts.simulation.runtime import run_causal_replan_step107_model as a; "
        "p=a.bind(); assert p.load_inputs is a.previous.load_inputs and p.request_worker is a.previous.request_worker; "
        "assert p.OUTPUT == a.OUTPUT and p.SESSION == a.SESSION and p.GPU == a.FIXED['gpu_uuid']"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
