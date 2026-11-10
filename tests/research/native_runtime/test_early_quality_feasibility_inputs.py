"""The quality audit has exactly the registered development-state matrix."""

import subprocess
import sys

import pytest

from scripts.tracking.trajectory_search import audit_early_quality_feasibility as quality


def test_exact_two_cases_and_eight_offsets_are_registered():
    assert len(quality.BRANCHES) == 16
    assert quality.branch_key(2, 0) == "case_02_offset_00"
    assert quality.branch_key(6, 7) == "case_06_offset_07"
    for case, offset in ((0, 0), (2, 8), (6, -1)):
        with pytest.raises(ValueError, match="Fixed saved"):
            quality.branch_key(case, offset)


def test_registered_software_gl_loads_before_mujoco_without_model():
    script = """\
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as q
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
prefix.environment()
inputs = q.load_inputs()
assert q.preload_software_gl(inputs['cases'][0], inputs['source_sha256']) == q.OSMESA
import mujoco
prefix.environment()
assert mujoco.__version__
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=quality.REPO, capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
