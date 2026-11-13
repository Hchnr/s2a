"""Verify mutable simulator positions are corrected from the saved state."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime import audit_bowl_predicate_alias_20261113 as audit
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def test_frozen_conditions_detach_both_reused_simulator_buffers():
    bowl, plate = np.asarray([0.0, 0.0, 0.1]), np.asarray([0.0, 0.0, 0.0])
    legacy = audit.original.on_conditions(bowl, plate, [], xy_limit=0.03)
    frozen = audit.frozen_conditions(bowl, plate, [], xy_limit=0.03)
    bowl[0], plate[0] = 0.01, 0.02
    assert legacy["bowl_world_m"][0] == 0.01 and legacy["plate_world_m"][0] == 0.02
    assert frozen["bowl_world_m"][0] == frozen["plate_world_m"][0] == 0.0


def test_free_joint_layout_uses_recorded_joint_order_and_rejects_nested_objects():
    xml = (
        '<mujoco><worldbody><body name="robot"><joint name="j" type="hinge"/></body>'
        '<body name="bowl"><freejoint/></body><body name="plate"><joint type="free"/></body>'
        "</worldbody></mujoco>"
    )
    assert audit.free_joint_positions(xml, body_names=("bowl", "plate"), expected_nq=15) == {"bowl": 1, "plate": 8}
    with pytest.raises(ValueError, match="model"):
        audit.free_joint_positions(xml, body_names=("bowl", "plate"), expected_nq=14)
    nested = xml.replace('<body name="plate">', '<body name="parent"><body name="plate">').replace(
        "</body></worldbody>", "</body></body></worldbody>"
    )
    with pytest.raises(ValueError, match="top-level"):
        audit.free_joint_positions(nested, body_names=("bowl", "plate"), expected_nq=15)


@pytest.fixture(scope="module")
def saved():
    inputs = audit.original.load_inputs()
    worker = prefix.read(audit.WORKER)
    addresses = audit.free_joint_positions(
        inputs["case"]["record_xml"],
        body_names=(inputs["config"]["bowl_body"], inputs["config"]["plate_body"]),
        expected_nq=41,
    )
    return inputs, worker["diagnosis"]["rows"], addresses


def compare(inputs, rows, addresses):
    return audit.compare_rows(
        rows,
        inputs["adaptive"],
        bowl_qpos=addresses[inputs["config"]["bowl_body"]],
        plate_qpos=addresses[inputs["config"]["plate_body"]],
        nq=41,
        nv=37,
        xy_limit=0.03,
        bowl_geoms=set(inputs["bowl_geoms"]),
        plate_geoms=set(inputs["plate_geoms"]),
    )


def test_real_saved_rows_recover_the_position_scalars_and_record_the_alias(saved):
    inputs, rows, addresses = saved
    assert addresses == {"akita_black_bowl_1_main": 9, "plate_1_main": 30}
    result = compare(inputs, rows, addresses)
    assert result["position_scalars_match_raw_states"] and not result["original_plate_vectors_valid"]
    assert result["original_plate_max_absolute_error_m"] == pytest.approx(0.0019539464728098865)
    assert result["all_native_labels_match"] and result["native_failed_endpoints"] == [129]
    step = result["corrected_rows"][21]
    assert step["endpoint_step"] == 129 and step["height_condition"] and step["xy_condition"]
    assert not step["contact_condition"] and step["contact_count"] == 0


@pytest.mark.parametrize("change", ["scalar", "contact", "label", "state_hash"])
def test_corrupted_saved_diagnosis_is_not_reinterpreted(saved, change):
    inputs, original, addresses = saved
    rows = copy.deepcopy(original)
    if change == "scalar":
        rows[0]["xy_distance_m"] += 0.01
    elif change == "contact":
        rows[0]["contact_count"] += 1
    elif change == "label":
        rows[0]["native_success"] = not rows[0]["native_success"]
    else:
        rows[0]["state_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        compare(inputs, rows, addresses)
