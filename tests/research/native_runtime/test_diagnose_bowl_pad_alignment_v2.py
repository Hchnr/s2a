"""Recovered pad snapshots must not alias a mutable MuJoCo data buffer."""

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime import diagnose_bowl_pad_alignment_v2 as recovery


def test_recovery_config_preserves_failed_cost_and_fixes_snapshot_ownership(monkeypatch):
    config = json.loads(recovery.CONFIG.read_text())
    monkeypatch.setattr(recovery.previous.recovery.previous, "FIXED", recovery.FIXED)
    recovery.previous.recovery.previous.validate_config(config)
    for key, value in (
        ("failed_alignment_forwards", 0),
        ("before_recovery_static_forwards", 204),
        ("snapshot_ownership", "view"),
        ("max_forwards", 98),
    ):
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError, match="saved-geometry scope"):
            recovery.previous.recovery.previous.validate_config(changed)


def test_admission_binds_failure_and_prior_cost_without_loading_mujoco():
    code = (
        "import sys; from scripts.simulation.runtime import diagnose_bowl_pad_alignment_v2 as s; "
        "e=s.bind(); x=e.load_inputs(verify_assets=False); "
        "assert x['refs']['snapshot_failure_audit']==s.FAILURE; "
        "assert x['config']['before_recovery_static_forwards']==301; "
        "assert e.reconstruct_endpoint is s.reconstruct_endpoint and 'mujoco' not in sys.modules"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_real_mujoco_previous_snapshot_survives_later_forward(monkeypatch):
    import mujoco

    xml = """<mujoco><worldbody>
      <body name="object"><freejoint/><geom name="target" type="box" size=".1 .1 .1" mass="1"/></body>
      <body name="l" pos="-.12 0 0"><geom name="left" type="box" size=".05 .05 .05"/><site name="site"/></body>
      <body name="r" pos=".12 0 0"><geom name="right" type="box" size=".05 .05 .05"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    monkeypatch.setattr(
        recovery.previous, "FIXED", dict(left_pad="left", right_pad="right", target_contact_geoms=["target"])
    )
    kwargs = dict(
        site_id=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "site")),
        body_id=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")),
        site_from_tcp=np.eye(4),
        forward=mujoco.mj_forward,
    )
    state = np.r_[0.0, model.qpos0, np.zeros(model.nv)]
    first = recovery.reconstruct_endpoint(model, data, state, **kwargs)
    stored = copy.deepcopy(first)
    assert not np.shares_memory(first["pad_alignment"]["left_pad_world_m"], data.geom_xpos)
    assert not np.shares_memory(first["pad_alignment"]["body_origin_world_m"], data.xpos)
    assert not np.shares_memory(first["pad_alignment"]["site_rotation_world"], data.site_xmat)
    state[1] += 0.04
    lbody = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l"))
    model.body_pos[lbody, 0] -= 0.03
    second = recovery.reconstruct_endpoint(model, data, state, **kwargs)
    for key in (
        "left_pad_world_m",
        "right_pad_world_m",
        "body_origin_world_m",
        "site_position_world_m",
        "site_rotation_world",
        "body_from_site_in_site_m",
    ):
        np.testing.assert_array_equal(first["pad_alignment"][key], stored["pad_alignment"][key])
    assert not np.array_equal(first["pad_alignment"]["left_pad_world_m"], second["pad_alignment"]["left_pad_world_m"])
    assert not np.array_equal(
        first["pad_alignment"]["body_origin_world_m"], second["pad_alignment"]["body_origin_world_m"]
    )
    assert data.time == 0
