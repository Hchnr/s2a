"""Audit rejects a changed model reference and counts active contact samples."""

import pytest

from scripts.verification.response_models.audit_box_face_feedback import count_active, same_reference


def test_reference_identity_includes_image_response_and_all_targets():
    original = dict(
        model_state_sha256="state",
        model_image_ref={"sha256": "image"},
        model_response_ref={"sha256": "response"},
        original_targets=[[[1, 0], [0, 1]]],
    )
    same_reference(original, original)
    changed = {**original, "model_image_ref": {"sha256": "other"}}
    with pytest.raises(ValueError):
        same_reference(original, changed)
    changed = {**original, "original_targets": [[[1, 0], [0, 0]]]}
    with pytest.raises(AssertionError):
        same_reference(original, changed)


def test_contact_audit_counts_samples_instead_of_contact_points():
    contact = dict(
        active_constraint=True, kind="robot_environment", geom1="table_collision", geom2="gripper0_finger2_collision"
    )
    inactive = {**contact, "active_constraint": False}
    samples = [{"contacts": [contact, contact]}, {"contacts": [inactive]}, {"contacts": []}]
    assert count_active(samples) == 1
    assert count_active(samples, {"table_collision", "gripper0_finger2_collision"}) == 1
