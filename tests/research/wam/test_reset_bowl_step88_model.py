"""The shared single-image owner must receive the verified step-88 observation."""

import copy
import hashlib
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import run_reset_bowl_step88_model as owner


def valid_capture():
    snapshot = dict(step=88, sim_time_s=4.65, state_sha256=owner.STATE_SHA)
    binding = {"sim_time": snapshot["sim_time_s"], "sim_state_sha256": owner.STATE_SHA}
    return dict(
        complete=True,
        passed=True,
        sources_unchanged=True,
        control_qualified=False,
        counts=dict(control_steps=88, native_steps=2200, stop=None),
        independent_max_absolute_difference={"command": 0.0, "state": 0.0},
        terminal_capture=dict(
            snapshot=snapshot,
            capture=dict(
                frame_index=88,
                current_endpoint={"endpoint_step": 88, **binding},
                render_before=dict(binding),
                render_after=dict(binding),
                future_rgb_frames=0,
                canvas_png={"path": "unused.png", "sha256": owner.IMAGE_SHA},
                camera_roles=["auxiliary_0", "left_wrist"],
                canvas_shape=[960, 640, 3],
            ),
        ),
    )


def test_accepts_the_replayed_step88_capture():
    initial, image = owner.validate_capture(valid_capture())
    assert initial == dict(step=88, sim_time_s=4.65, state_sha256=owner.STATE_SHA)
    assert image["canvas_png"]["sha256"] == owner.IMAGE_SHA


@pytest.mark.parametrize("mutation", ["step", "state", "image", "future", "physics", "difference"])
def test_rejects_a_mismatched_capture_before_model_work(mutation):
    result = valid_capture()
    if mutation == "step":
        result["terminal_capture"]["snapshot"]["step"] = 80
    elif mutation == "state":
        result["terminal_capture"]["capture"]["render_after"]["sim_state_sha256"] = "a" * 64
    elif mutation == "image":
        result["terminal_capture"]["capture"]["canvas_png"]["sha256"] = "a" * 64
    elif mutation == "future":
        result["terminal_capture"]["capture"]["future_rgb_frames"] = 1
    elif mutation == "physics":
        result["counts"]["control_steps"] = 87
    else:
        result["independent_max_absolute_difference"]["state"] = 0.01
    with pytest.raises(ValueError, match="step-88 image and state"):
        owner.validate_capture(result)


def test_stages_the_exact_image_with_a_step_neutral_request_path(tmp_path, monkeypatch):
    directory = tmp_path / "runtime"
    (directory / "anchor0048").mkdir(parents=True)
    data = b"actual current image"
    (directory / "anchor0048/current.png").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(owner, "IMAGE_SHA", digest)
    image = owner.stage_current_image(directory)
    assert image == dict(path="current.png", sha256=digest)
    assert (directory / image["path"]).read_bytes() == data
    with pytest.raises(FileExistsError):
        owner.stage_current_image(directory)


def test_redirects_only_the_registered_workers_without_mutating_specs(monkeypatch):
    seen = []
    monkeypatch.setattr(
        owner, "_OWNED_SOCKET_PAIR", lambda specs, *, timeout_seconds: seen.append((specs, timeout_seconds))
    )
    specs = {
        "simulation": {"argv": ["python", "-m", owner.OLD_MODULE, "--worker", "request"]},
        "model": {"argv": ["python", "-m", owner.OLD_MODULE, "--worker", "model"]},
    }
    original = copy.deepcopy(specs)
    owner.redirect_workers(specs, timeout_seconds=123)
    assert specs == original and seen[0][1] == 123
    assert all(row["argv"][2] == owner.MODULE for row in seen[0][0].values())
    specs["model"]["argv"][2] = "unrelated.module"
    with pytest.raises(ValueError, match="registered model-owner workers"):
        owner.redirect_workers(specs, timeout_seconds=123)


def test_binding_keeps_the_original_model_owner_source_unchanged():
    code = (
        "from scripts.simulation.runtime import run_reset_bowl_step88_model as a; "
        "p=a.bind(); assert p.load_inputs is a.load_inputs and p.request_worker is a.request_worker; "
        "assert p.OUTPUT == a.OUTPUT and p.SESSION == a.SESSION"
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
