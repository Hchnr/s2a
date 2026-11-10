"""Meaningful geometry and causal-reference checks for the offline diagnostic."""

import mujoco
import numpy as np
import pytest

from s2a.research.wam.tracking.preview.development_reference_geometry import (
    StaticReferenceProbe,
    contact_flags,
    first_closure,
    migrate_xml,
)
from scripts.wam.tracking.preview.diagnose_development_reference_geometry import expected_case, safe_ref


def test_migration_changes_only_registered_file_attributes_and_reverses():
    raw = b'<mujoco><asset><mesh name="a" file="/old/a"/><texture file="/old/b"/></asset></mujoco>'
    migrated = migrate_xml(raw, {"/old/a": "/safe/a", "/old/b": "/safe/b"})
    assert migrated == raw.replace(b"/old/a", b"/safe/a").replace(b"/old/b", b"/safe/b")
    with pytest.raises(ValueError, match="declarations"):
        migrate_xml(raw, {"/old/a": "/safe/a"})


def test_first_generated_close_uses_strict_threshold_and_full_chunk():
    openings = np.ones(40)
    openings[4] = 0.35
    openings[17] = 0.349
    assert first_closure(openings) == 17
    openings[17] = 0.35
    assert first_closure(openings) is None
    openings[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        first_closure(openings)


def test_contacts_require_target_pair_and_nonpositive_distance():
    assert contact_flags([(2, 10, -0.001), (10, 3, 0.0), (4, 11, -1)], {10}, [2, 3]) == [True, True]
    assert contact_flags([(2, 10, 0.001), (3, 11, -1)], {10}, [2, 3]) == [False, False]
    with pytest.raises(ValueError, match="Distinct"):
        contact_flags([], {10}, [2, 2])


def test_fixed_case_matrix_and_external_evidence_rejection():
    assert expected_case(0) == ("task0_init0_s42_raw", 8)
    assert expected_case(7) == ("task1_init0_s43_terminal058", 9)
    assert expected_case(11) == ("task2_init0_s43_terminal058", 6)
    with pytest.raises(ValueError, match="lexically inside"):
        safe_ref("/not-the-repository/old-evidence.json")


def test_identity_and_reference_relocation_preserve_object_and_budget():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <body name="robot0_base">
            <site name="gripper0_grip_site"/>
            <body pos="-.024 0 0"><joint name="gripper0_finger_joint1" type="slide" axis="1 0 0"/>
              <geom name="gripper0_finger1_pad_collision" type="box" size=".01 .01 .01"/></body>
            <body pos=".024 0 0"><joint name="gripper0_finger_joint2" type="slide" axis="1 0 0"/>
              <geom name="gripper0_finger2_pad_collision" type="box" size=".01 .01 .01"/></body>
          </body>
          <body name="target_body"><freejoint/><geom name="target_collision" type="sphere" size=".02"/></body>
        </worldbody></mujoco>
        """
    )
    probe = StaticReferenceProbe(model, "target", budget=3)
    state = np.r_[0.0, model.qpos0, np.zeros(model.nv + model.na)]
    probe.restore(state)
    assert probe.contacts() == [True, True]
    current = probe.site_transform()
    probe.place(current)
    assert probe.contacts() == [True, True]
    moved = current.copy()
    moved[1, 3] += 0.1
    probe.place(moved)
    assert probe.contacts() == [False, False]
    assert probe.queries == 3
    with pytest.raises(RuntimeError, match="budget"):
        probe.forward()
