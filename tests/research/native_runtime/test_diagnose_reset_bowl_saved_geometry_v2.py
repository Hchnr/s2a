"""Recovery must preserve the failed query and bind the normalization guard."""

import copy
import json
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import diagnose_reset_bowl_saved_geometry_v2 as recovery


def test_recovery_config_records_spent_work_and_strict_tolerance(monkeypatch):
    config = json.loads(recovery.CONFIG.read_text())
    monkeypatch.setattr(recovery.previous, "FIXED", recovery.FIXED)
    recovery.previous.validate_config(config)
    for key, value in (
        ("previous_failed_forwards", 0),
        ("max_forwards", 98),
        ("max_quaternion_component_change", 1e-6),
        ("physics_steps", 1),
    ):
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError, match="saved-geometry scope"):
            recovery.previous.validate_config(changed)


def test_recovery_binds_original_failure_and_sources_in_a_new_process():
    code = (
        "from scripts.simulation.runtime import diagnose_reset_bowl_saved_geometry_v2 as s; "
        "e=s.bind_recovery(); x=e.load_inputs(verify_assets=False); "
        "assert e.OUTPUT == s.OUTPUT and e.MODULE == s.MODULE; "
        "assert x['refs']['prior_failure'] == s.FAILURE; "
        "assert x['config']['previous_failed_forwards'] == 10; "
        "assert e.reconstruct_endpoint is s.reconstruct_endpoint"
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_bound_geometry_counts_97_queries_and_preserves_step_guard():
    code = """
import sys, numpy as np
from scripts.simulation.runtime import diagnose_reset_bowl_saved_geometry_v2 as s
from tests.research.native_runtime.test_diagnose_reset_bowl_saved_geometry import fake_inputs, fake_mujoco, counters
m=fake_mujoco(); original_factory=m.MjModel.from_xml_string
class Model:
    @staticmethod
    def from_xml_string(text):
        value=original_factory(text); value.njnt=0; value.jnt_type=np.array([]); value.jnt_qposadr=np.array([]); return value
m.MjModel=Model
sys.modules['mujoco']=m
engine=s.bind_recovery(); count=counters(); result=engine.run_geometry(fake_inputs(),count,mujoco=m)
assert count['models']==count['data']==1 and count['forwards_completed']==97 and count['physics_steps']==0
assert result['normalization_summary']['queries']==97 and result['normalization_summary']['queries_with_change']==0
"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
