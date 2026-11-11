"""A failed worker must retain its own evidence and avoid unadmitted work."""

import json
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import reset_multitask_live_admission as admission
from scripts.simulation.runtime import reset_multitask_live_workers as workers


def reject(*_args):
    raise ValueError("synthetic admission failure")


def test_failed_model_admission_records_zero_initializations(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "OUTPUT", tmp_path)
    directory = workers.case_directory("producer", 4) / "model"
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="synthetic admission failure"):
        workers.model_worker(4, reject)
    result = json.loads((directory / "result.json").read_text())
    assert result["case_index"] == 4 and result["gpu_uuid"] == admission.CASES[1]["gpu_uuid"]
    assert result["model_initializations"] == result["model_requests_attempted"] == 0
    assert result["complete"] is result["passed"] is False
    assert result["error"]["message"] == "synthetic admission failure"


def test_failed_checker_admission_is_retained_without_simulation(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "OUTPUT", tmp_path)
    directory = workers.case_directory("checker", 8) / "simulation"
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="synthetic admission failure"):
        workers.simulation_worker(8, "checker", reject)
    result = json.loads((directory / "result.json").read_text())
    assert result["case_index"] == 8 and result["phase"] == "checker"
    assert result["complete"] is result["passed"] is False
    assert "runtime" not in result and result["error"]["message"] == "synthetic admission failure"


def test_worker_module_does_not_import_candidate_or_simulator():
    code = (
        "from scripts.simulation.runtime import reset_multitask_live_workers as s; "
        "s.isolation.checker_isolation(); s.original.require_no_simulator_import()"
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("phase", "index"), [("producer", 1), ("checker", 2), ("unknown", 0)])
def test_worker_rejects_unregistered_cases(phase, index):
    with pytest.raises(ValueError, match="registered phase and case"):
        workers.case_directory(phase, index)
