"""The diagnostic must retain exact inputs, scope and prior execution cost."""

import copy
import json
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import diagnose_reset_bowl_saved_contacts as diagnosis


def test_contact_config_rejects_changed_budget_geometry_and_failure_history(monkeypatch):
    config = json.loads(diagnosis.CONFIG.read_text())
    monkeypatch.setattr(diagnosis.recovery.previous, "FIXED", diagnosis.FIXED)
    diagnosis.recovery.previous.validate_config(config)
    for key, value in (
        ("max_forwards", 98),
        ("prior_recovery_forwards", 0),
        ("physics_steps", 1),
        ("left_pad", "other"),
        ("target_contact_geoms", config["target_contact_geoms"][:-1]),
    ):
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError, match="saved-geometry scope"):
            diagnosis.recovery.previous.validate_config(changed)


def test_bound_inputs_verify_xml_and_prior_results_without_loading_mujoco():
    code = (
        "import sys; from scripts.simulation.runtime import diagnose_reset_bowl_saved_contacts as s; "
        "e=s.bind_contacts(); x=e.load_inputs(verify_assets=False); "
        "assert e.OUTPUT == s.OUTPUT and e.MODULE == s.MODULE; "
        "assert e.reconstruct_endpoint is s.reconstruct_endpoint; "
        "assert x['refs']['prior_recovery'] == s.PRIOR; "
        "assert len(x['config']['target_contact_geoms']) == 40; "
        "assert len(x['prior_geometry']['points']['fixed']) == 48; "
        "assert 'mujoco' not in sys.modules"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_xml_rejects_visual_geom_in_contact_group():
    config = dict(body_name="bowl", target_contact_geoms=["g1"], left_pad="l", right_pad="r")
    xml = (
        "<mujoco><worldbody><body name='bowl'><geom name='akita_black_bowl_1_g0' group='1'/>"
        "<geom name='g1'/></body><geom name='l'/><geom name='r'/></worldbody></mujoco>"
    )
    diagnosis.check_xml(xml, config)
    with pytest.raises(ValueError, match="collision and visual"):
        diagnosis.check_xml(xml.replace("group='1'", "group='0'"), config)
