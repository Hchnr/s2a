"""Check long-run budgets and the independently replayed prior evidence."""

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.simulation.runtime import adaptive_live_bowl_long_admission as admission
from scripts.simulation.runtime import adaptive_live_bowl_recovery_admission as previous
from scripts.simulation.runtime import registered_model_bundle as registered
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def configs():
    return json.loads(admission.CONFIG.read_text()), json.loads(previous.CONFIG.read_text())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("case_id", "another_case"),
        ("session", "old_session"),
        ("max_model_requests", 39),
        ("max_adaptive_periods", 8),
        ("max_native_steps_per_phase", 3674),
        ("max_denoise_calls", 1399),
        ("automatic_retry", True),
        ("fresh_cases", False),
        ("prior_checker_sha256", "0" * 64),
        ("previous_physical_bounds", {"control_steps": {"lower": 1761, "upper": 1761}}),
    ],
)
def test_changed_long_registration_is_rejected(key, value):
    config, old = configs()
    admission.validate_config(config, old)
    config[key] = value
    with pytest.raises(ValueError, match="identity, prior ranges and budgets"):
        admission.validate_config(config, old)


@pytest.mark.parametrize("phase", ["producer", "checker"])
def test_phase_registration_binds_new_session_and_capture_root(phase):
    config, _ = configs()
    inputs = {"config": config, "initial": {"step": 107}}
    capture = admission.OUTPUT / phase / "simulation"
    result = admission.model_registration(inputs, capture)
    assert result["session"] == admission.SESSION
    assert result["image_root"] == str(capture.relative_to(admission.ROOT))
    assert result["max_requests"] == result["max_periods"] == 40
    with pytest.raises(ValueError, match="capture root"):
        admission.model_registration(inputs, previous.OUTPUT / phase / "simulation")


@pytest.fixture(scope="module")
def real_history():
    inputs = previous.load_inputs()
    producer = prefix.read(
        {"path": str(previous.OUTPUT / "producer/result.json"), "sha256": admission.PRIOR_PRODUCER_SHA}
    )
    checker = prefix.read({"path": str(previous.OUTPUT / "checker/result.json"), "sha256": admission.PRIOR_CHECKER_SHA})
    return inputs, producer, checker, admission.previous_owner.source_refs()


@pytest.mark.parametrize("change", ["task_count", "cumulative", "registration", "unreaped"])
def test_prior_checker_tampering_is_rejected(real_history, monkeypatch, change):
    inputs, producer, original, sources = real_history
    checker = copy.deepcopy(original)
    read = prefix.read
    if change == "task_count":
        checker["counts"]["task_success_periods"] += 1
    elif change == "cumulative":
        checker["cumulative_physical_bounds"]["control_steps"]["upper"] -= 1
    else:
        target = checker["registration_ref"] if change == "registration" else checker["workers_ref"]

        def altered(ref, seen=None):
            value = read(ref, seen)
            if ref == target:
                if change == "registration":
                    value["source_refs"] = []
                else:
                    value["workers"][0]["reaped"] = False
            return value

        monkeypatch.setattr(prefix, "read", altered)
    with pytest.raises(ValueError):
        admission.verify_previous(inputs, producer, checker, sources)


def test_actual_history_and_model_python_agree_without_loading_gpu_or_simulator():
    inputs = admission.load_inputs()
    expected = registered._digest(admission.input_receipt(inputs))
    script = """
import sys
from scripts.simulation.runtime import adaptive_live_bowl_long_admission as admission
from scripts.simulation.runtime import registered_model_bundle as registered
inputs = admission.load_inputs()
assert not any(name in sys.modules for name in ('torch', 'mujoco', 'robosuite', 'libero', 'cosmos_framework'))
print(registered._digest(admission.input_receipt(inputs)))
"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    environment["PYTHONPATH"] = "src:."
    environment.update({key: "1" for key in prefix.THREAD_VARS})
    result = subprocess.run(
        [inputs["prepared"]["model_config"]["model_python"], "-c", script],
        cwd=admission.ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.stdout.strip() == expected
    assert inputs["previous_physical_bounds"] == inputs["config"]["previous_physical_bounds"]
    assert inputs["previous_model_bounds"] == admission.previous_owner.known_model_bounds(inputs["config"])
    assert Path(inputs["refs"]["prior_checker"]["path"]) == previous.OUTPUT / "checker/result.json"
