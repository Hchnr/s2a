"""The precise solver must handle the recorded bad-residual pair without changing gates."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from s2a.research.native_runtime.bilateral_box_feasibility import solve_infinity
from scripts.simulation.runtime import analyze_bowl_static_close_feasibility_v2 as recovery


def recorded_problem():
    value = json.loads(Path(recovery.DIAGNOSTIC["path"]).read_text())
    a, b = np.asarray(value["constraint_matrix"]), np.asarray(value["constraint_bound"])
    assert a.shape == (66, 5)
    left, right = (a[:30, :4], b[:30]), (a[30:60, :4], b[30:60])
    error = b[-6::2]
    return left, right, error, value["max_closing_m"]


def test_recorded_bad_residual_passes_without_relaxing_constraint_check():
    args = recorded_problem()
    with pytest.raises(ValueError, match="constraint-verified"):
        solve_infinity(*args)
    result = recovery.solve_precise(*args)
    assert result["success"] and result["maximum_constraint_violation_m"] < 1e-8
    assert result["infinity_error_m"] == pytest.approx(0.05731103746050923, abs=1e-10)


def test_precise_tolerances_are_explicitly_sent_to_highs(monkeypatch):
    original, captured = recovery.linprog, []

    def record(*args, **kwargs):
        captured.append(kwargs["options"])
        return original(*args, **kwargs)

    monkeypatch.setattr(recovery, "linprog", record)
    recovery.solve_precise(*recorded_problem())
    assert captured == [dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9)]


def test_recovery_binds_original_failure_and_sources_without_simulator():
    code = (
        "import sys; from scripts.simulation.runtime import analyze_bowl_static_close_feasibility_v2 as s; "
        "p=s.bind(); assert p.solve_infinity is s.solve_precise and p.OUTPUT==s.OUTPUT; "
        "assert p.FIXED['prior_lp_attempts']==13 and 'mujoco' not in sys.modules"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
