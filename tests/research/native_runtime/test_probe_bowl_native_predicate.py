"""Exercise native predicate decomposition without advancing a simulator."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.simulation.runtime import probe_bowl_native_predicate as probe


@pytest.mark.parametrize(
    ("bowl", "plate", "pairs", "expected"),
    [
        ([0, 0, 1], [0, 0, 0], [{"pair": 1}], (True, True, True, True)),
        ([0.03, 0, 1], [0, 0, 0], [{"pair": 1}], (True, True, False, False)),
        ([0, 0, -1], [0, 0, 0], [{"pair": 1}], (False, True, True, False)),
        ([0, 0, 1], [0, 0, 0], [], (True, False, True, False)),
        ([0, 0, 0], [0, 0, 0], [{"pair": 1}], (True, True, True, True)),
    ],
)
def test_native_on_conjunction_and_strict_xy_boundary(bowl, plate, pairs, expected):
    row = probe.on_conditions(bowl, plate, pairs, xy_limit=0.03)
    assert tuple(row[key] for key in ("height_condition", "contact_condition", "xy_condition", "reconstructed_on")) == (
        expected
    )


def test_invalid_positions_and_collision_groups_are_rejected():
    with pytest.raises(ValueError):
        probe.on_conditions([0, 0, np.nan], [0, 0, 0], [], xy_limit=0.03)
    xml = '<mujoco><worldbody><body name="b"><geom name="a"/><geom name="v" group="1"/>'
    xml += '<body name="nested"><geom name="c" group="0"/><geom name="other" group="2"/></body></body></worldbody></mujoco>'
    assert probe.contact_geom_names(xml, "b") == ["a", "c"]
    with pytest.raises(ValueError):
        probe.contact_geom_names(xml, "missing")
    with pytest.raises(ValueError):
        probe.contact_geom_names(xml.replace('name="c"', 'name="a"'), "b")


@pytest.mark.parametrize("change", [None, "missing", "duplicate", "failed", "unreaped"])
def test_actual_synthetic_qa_model_and_forward_are_counted_once(change):
    report = dict(nodeid=probe.SYNTHETIC_NODE, when="call", outcome="passed", wasxfail=None)
    qa = dict(
        complete=True,
        passed=True,
        all_tests_passed=True,
        actual_wait=True,
        reaped=True,
        source_unchanged=True,
        configured_parallelism=64,
        execution_parallelism=64,
        signal_sent=False,
        workers=[dict(test_reports=[report])],
    )
    if change == "missing":
        qa["workers"][0]["test_reports"].clear()
    elif change == "duplicate":
        qa["workers"][0]["test_reports"].append(copy.deepcopy(report))
    elif change == "failed":
        report["outcome"] = "failed"
    elif change == "unreaped":
        qa["reaped"] = False
    if change is None:
        assert probe.verify_synthetic_qa(qa, parallelism=64) == 1
    else:
        with pytest.raises(ValueError, match="synthetic MuJoCo test"):
            probe.verify_synthetic_qa(qa, parallelism=64)


def synthetic_inputs():
    eye = np.eye(4).tolist()
    config = dict(
        model_shape=dict(nq=41, nv=37, na=0),
        tcp_site="site",
        bowl_body="bowl_body",
        plate_body="plate_body",
        max_forwards=40,
        xy_limit_m=0.03,
        tcp_anchor_tolerance_m=1e-9,
    )
    periods = [
        dict(
            step=step - 1,
            actual_state=[0] * 79,
            endpoint_receipt={"sim_state_sha256": "recorded"},
            task_success=True,
            target_grasp=False,
        )
        for step in range(108, 148)
    ]
    captures = [dict(frame_index=step, current_source_pose=eye) for step in (107, 111, 119, 120, 128, 136, 144)]
    case = dict(record_xml="synthetic", expected=dict(controller_site_pose=eye, source_pose=eye), tool=eye)
    return dict(
        config=config,
        case=case,
        bowl_geoms=["bowl_geom"],
        plate_geoms=["plate_geom"],
        adaptive=periods,
        branch={"runtime": {"adaptive_checker_captures": captures}},
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


def fake_mujoco(*, attempt_step=False):
    model = SimpleNamespace(nq=41, nv=37, na=0)
    data = SimpleNamespace(
        xpos=np.asarray([[0.0, 0.0, 0.1], [0.0, 0.0, 0.0]]),
        ncon=1,
        contact=[SimpleNamespace(geom1=0, geom2=1, dist=-0.001)],
    )
    fake = SimpleNamespace(
        __version__="synthetic",
        mjtObj=SimpleNamespace(mjOBJ_SITE=1, mjOBJ_BODY=2, mjOBJ_GEOM=3),
        MjModel=SimpleNamespace(from_xml_string=lambda _xml: model),
        MjData=lambda _model: data,
        mj_name2id=lambda _model, kind, name: {
            (1, "site"): 0,
            (2, "bowl_body"): 0,
            (2, "plate_body"): 1,
            (3, "bowl_geom"): 0,
            (3, "plate_geom"): 1,
        }.get((kind, name), -1),
        mj_id2name=lambda _model, _kind, index: ["bowl_geom", "plate_geom"][index],
        mj_step=lambda *_args: None,
        mj_step1=lambda *_args: None,
        mj_step2=lambda *_args: None,
    )

    def forward(_model, _data):
        if attempt_step:
            fake.mj_step(_model, _data)

    fake.mj_forward = forward
    return fake


def stub_geometry(_model, data, _state, *, forward, **_kwargs):
    forward(_model, data)
    return dict(
        tcp_world_m=np.zeros(3),
        body_origin_world_m=data.xpos[0].copy(),
        state_change=dict(changed_indices=[], max_absolute_change=0.0, quaternion_norms=[]),
    )


def test_label_mismatch_is_reported_without_becoming_a_valid_explanation(monkeypatch):
    inputs, work, fake = synthetic_inputs(), counters(), fake_mujoco()
    inputs["adaptive"][21]["task_success"] = False
    monkeypatch.setattr(probe, "state_sha256", lambda _state: "recorded")
    monkeypatch.setattr(probe, "reconstruct_endpoint", stub_geometry)
    result = probe.query_saved_states(inputs, work, mujoco=fake)
    assert result["label_mismatch_endpoints"] == [129]
    assert result["all_native_labels_match"] is result["interpretation_eligible"] is False
    assert result["native_failed_endpoints"] == [129] and result["reconstructed_failed_endpoints"] == []
    assert work["models"] == work["data"] == 1
    assert work["forwards_attempted"] == work["forwards_completed"] == 40
    assert work["physics_steps"] == work["contexts"] == work["model_requests"] == 0


def test_forward_cannot_advance_physics_and_hooks_are_restored(monkeypatch):
    inputs, work, fake = synthetic_inputs(), counters(), fake_mujoco(attempt_step=True)
    originals = [fake.mj_step, fake.mj_step1, fake.mj_step2]
    monkeypatch.setattr(probe, "state_sha256", lambda _state: "recorded")
    monkeypatch.setattr(probe, "reconstruct_endpoint", stub_geometry)
    with pytest.raises(ValueError, match="cannot advance physics"):
        probe.query_saved_states(inputs, work, mujoco=fake)
    assert work["forwards_attempted"] == 1 and work["forwards_completed"] == 0
    assert [fake.mj_step, fake.mj_step1, fake.mj_step2] == originals


def test_actual_completed_replay_and_native_sources_are_admitted():
    inputs = probe.load_inputs()
    assert len(inputs["adaptive"]) == 40
    assert len(inputs["bowl_geoms"]) == 40 and len(inputs["plate_geoms"]) == 10
    assert inputs["adaptive"][21]["step"] + 1 == 129
    assert inputs["adaptive"][21]["task_success"] is False
    altered = copy.deepcopy(inputs["adaptive"][21])
    altered["actual_state"][0] += 0.01
    assert (
        probe.state_sha256(np.asarray(altered["actual_state"], dtype=np.float64))
        != altered["endpoint_receipt"]["sim_state_sha256"]
    )
