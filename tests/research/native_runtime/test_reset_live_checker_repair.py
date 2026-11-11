"""The repaired checker keeps the original evidence and a bounded replay."""

import json
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import run_reset_live_checker_repair as repair


def test_repair_registration_rejects_new_model_or_more_physical_steps():
    config = json.loads(repair.CONFIG.read_text())
    repair.validate_config(config)
    for key, value in (
        ("max_control_steps", 17),
        ("max_native_steps", 401),
        ("world_model_calls", 1),
        ("automatic_retry", True),
        ("case_index", 0),
    ):
        with pytest.raises(ValueError, match="exact repair"):
            repair.validate_config(dict(config, **{key: value}))


def test_repair_checker_import_is_candidate_and_simulator_free():
    code = "from scripts.simulation.runtime import run_reset_live_checker_repair as s; s.isolation.checker_isolation(); s.original.require_no_simulator_import()"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
