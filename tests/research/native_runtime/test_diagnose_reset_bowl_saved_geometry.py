"""Saved geometry is bounded and tied to the original physical observations."""

import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.simulation.runtime import diagnose_reset_bowl_saved_geometry as diagnosis


def test_config_rejects_changed_budget_or_target():
    config = json.loads(diagnosis.CONFIG.read_text())
    diagnosis.validate_config(config)
    for key, value in (
        ("max_models", 2),
        ("max_forwards", 98),
        ("body_name", "other"),
        ("physics_steps", 1),
        ("automatic_retry", True),
    ):
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError, match="saved-geometry scope"):
            diagnosis.validate_config(changed)


def test_real_input_binds_both_audited_trajectories_without_loading_mujoco():
    inputs = diagnosis.load_inputs(verify_assets=False)
    assert len(inputs["fixed_periods"]) == len(inputs["old_periods"]) == 48
    assert inputs["case"]["case"]["case_id"] == "task0_init0_s42_raw"
    assert inputs["xml_sha256"] == "95ff5b34197b55514415f1e0b4e2b6f8fe3a63d77b0ef50613b1a7ba198e322c"
    assert "mujoco" not in sys.modules


def fake_inputs():
    pose = np.eye(4)

    def state(step):
        return [step * 0.05, step * 0.01, 0.0, 0.0, 0.0]

    def capture(step):
        matrix = pose.copy()
        matrix[0, 3] = step * 0.01
        return dict(frame_index=step, current_source_pose=matrix)

    periods = [
        dict(
            step=i,
            actual_state=state(i + 1),
            endpoint_receipt=dict(sim_state_sha256="a" * 64),
            completed_feedback=dict(score=dict(opening=0.9)),
            target_grasp=False,
            task_success=False,
        )
        for i in range(48)
    ]
    return dict(
        config=copy.deepcopy(diagnosis.FIXED),
        case=dict(
            record_xml="synthetic",
            expected=dict(controller_site_pose=pose.copy(), source_pose=pose.copy()),
            tool=pose.copy(),
        ),
        initial=state(0),
        fixed_periods=copy.deepcopy(periods),
        old_periods=copy.deepcopy(periods),
        fixed=dict(runtime=dict(reset_render=capture(0), fixed_reference_capture=capture(8))),
        old=dict(runtime=dict(reset_render=capture(0), streaming_captures=[capture(i) for i in (8, 16, 24, 32, 40)])),
    )


def fake_mujoco():
    model = SimpleNamespace(nq=2, nv=2, na=0, nsite=1, nbody=1)

    class Model:
        @staticmethod
        def from_xml_string(text):
            assert text == "synthetic"
            return model

    class Data:
        def __init__(self, _model):
            self.time = 0.0
            self.qpos = np.zeros(2)
            self.qvel = np.zeros(2)
            self.site_xmat = np.eye(3).reshape(1, 9)
            self.site_xpos = np.zeros((1, 3))
            self.xpos = np.array([[0.2, 0, 0]])

    def forward(_model, data):
        data.site_xpos[0, 0] = data.qpos[0]

    def step(*_args):
        raise AssertionError("physics stepping must never be called")

    return SimpleNamespace(
        MjModel=Model,
        MjData=Data,
        mjtObj=SimpleNamespace(mjOBJ_SITE=1, mjOBJ_BODY=2),
        mj_name2id=lambda *_: 0,
        mj_forward=forward,
        mj_step=step,
        mj_step1=step,
        mj_step2=step,
    )


def counters():
    return dict(
        model_attempts=0,
        models=0,
        data_attempts=0,
        data=0,
        forwards_attempted=0,
        forwards_completed=0,
        physics_steps=0,
        contexts=0,
        model_requests=0,
    )


def test_geometry_queries_all_saved_endpoints_without_stepping(monkeypatch):
    mujoco, counts = fake_mujoco(), counters()
    original_step = mujoco.mj_step
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    result = diagnosis.run_geometry(fake_inputs(), counts, mujoco=mujoco)
    assert counts["models"] == counts["data"] == 1 and counts["forwards_completed"] == 97
    assert counts["physics_steps"] == counts["contexts"] == counts["model_requests"] == 0
    assert mujoco.mj_step is original_step and len(result["capture_checks"]) == 8
    assert result["summaries"]["fixed"]["minimum_distance_m"] == pytest.approx(0)
    assert result["summaries"]["fixed"]["minimum_distance_endpoint_step"] == 20


def test_geometry_stops_at_the_query_limit_and_restores_step_functions(monkeypatch):
    mujoco, counts, inputs = fake_mujoco(), counters(), fake_inputs()
    original_step = mujoco.mj_step
    inputs["config"]["max_forwards"] = 96
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    with pytest.raises(ValueError, match="query budget is exhausted"):
        diagnosis.run_geometry(inputs, counts, mujoco=mujoco)
    assert counts["forwards_attempted"] == counts["forwards_completed"] == 96
    assert mujoco.mj_step is original_step and counts["physics_steps"] == 0
