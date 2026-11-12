"""The diagnostic must bind both replayed branches and prior static cost."""

import copy
import json
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import diagnose_bowl_pad_alignment as diagnosis


def test_config_freezes_forward_budget_target_and_prior_cost(monkeypatch):
    config = json.loads(diagnosis.CONFIG.read_text())
    monkeypatch.setattr(diagnosis.recovery.previous, "FIXED", diagnosis.FIXED)
    diagnosis.recovery.previous.validate_config(config)
    for key, value in (("max_forwards", 98), ("prior_static_forwards", 0), ("left_pad", "wrong"), ("physics_steps", 1)):
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError, match="saved-geometry scope"):
            diagnosis.recovery.previous.validate_config(changed)


def test_admission_binds_audited_baseline_and_lead_without_loading_mujoco():
    code = (
        "import sys; from scripts.simulation.runtime import diagnose_bowl_pad_alignment as s; "
        "e=s.bind(); x=e.load_inputs(verify_assets=False); "
        "assert x['refs']['prior_contacts_audit']==s.CONTACTS and x['refs']['lead_audit']==s.LEAD; "
        "assert len(x['fixed_periods'])==len(x['old_periods'])==48; "
        "assert x['old_periods'][-1]['completed_feedback']['score']['opening']<0.7; "
        "assert 'mujoco' not in sys.modules"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
