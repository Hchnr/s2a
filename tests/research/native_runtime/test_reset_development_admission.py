"""The owner retains fixed budgets and isolates each worker's case identity."""

import copy
import json
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime.run_reset_development_admission import CONFIG, case_receipt, validate_config


def test_reset_budget_rejects_physics_retries_and_roster_changes():
    original = json.loads(CONFIG.read_text())
    validate_config(original)
    for key, value in (
        ("max_native_steps", 1),
        ("max_control_steps", False),
        ("max_models", 37),
        ("case_indices", [0]),
        ("automatic_retry", True),
    ):
        changed = dict(original, **{key: value})
        with pytest.raises(ValueError):
            validate_config(changed)


def test_worker_receipt_contains_only_the_registered_case():
    case1 = {"case": {"case_index": 1}, "original_source_sha256": {"one": "sha"}, "original_receipt_ref": {"path": "a"}}
    case2 = {"case": {"case_index": 2}, "original_source_sha256": {"two": "sha"}, "original_receipt_ref": {"path": "b"}}
    inputs = dict(
        cases=[case1, case2],
        asset_refs={"asset": "sha"},
        mapped_ref={"path": "m"},
        source_preflight_ref={"path": "p"},
        boot_ref={"path": "b"},
        migrated_ref={"path": "r"},
        study_inputs={},
    )
    original = copy.deepcopy(inputs)
    receipt = case_receipt(inputs, 2)
    assert len(receipt["cases"]) == 1 and receipt["cases"][0]["case"]["case_index"] == 2
    assert inputs == original
    with pytest.raises(ValueError, match="One original case"):
        case_receipt(inputs, 3)


def test_owner_import_keeps_simulator_unloaded_before_setup():
    code = (
        "import sys; from scripts.simulation.runtime import run_reset_development_admission; "
        "assert not any(name in sys.modules for name in ('mujoco', 'OpenGL', 'robosuite', 'libero'))"
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
