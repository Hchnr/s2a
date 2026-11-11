"""Audit uses completed poses and rejects changed target-contact labels."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import pose_error, transform
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.reference_ports import audit_reference_table_geometry as audit


def test_reconstruct_measured_pose_handles_noncommuting_rotations():
    measured = transform(np.array([0.1, -0.2, 0.3]), Rotation.from_euler("xyz", [0.4, -0.3, 0.2]).as_matrix())
    desired = transform(np.array([-0.1, 0.4, 0.25]), Rotation.from_euler("xyz", [-0.2, 0.1, 0.6]).as_matrix())
    recovered = audit.reconstruct_measured_tcp(desired, pose_error(desired, measured))
    np.testing.assert_allclose(recovered, measured, atol=1e-12, rtol=0)


def test_audit_binds_all_geometry_and_physical_history():
    result = audit.audit()
    summary = result["summary"]
    assert summary["target_intersection_steps"] == list(range(120, 127))
    assert summary["executed_target_intersections"] == 6 and summary["stopped_target_intersection"] is True
    assert summary["prestate_contact_disagreements_with_previous_solver_cache"] == []
    assert summary["independent_current_tcp_error_max_difference"] <= 1e-12
    assert result["audit_forward_queries"] == result["worker_physics_steps"] == 0
    assert result["inverse_kinematics_verified"] is result["control_qualified"] is False


def test_mislabeled_target_contact_is_rejected():
    inputs = audit.geometry.load_inputs()
    owner = prefix.read({"path": str(audit.RUN / "result.json"), "sha256": audit.OWNER_SHA})
    branch = prefix.read(owner["branch_ref"])
    force_owner = prefix.read(audit.FORCE_OWNER)
    force_branch = prefix.read(force_owner["summaries"]["historical"]["branch_ref"])
    altered = copy.deepcopy(branch)
    altered["analysis"]["periods"][40]["target_proxy_finger_table_intersection"] = False
    with pytest.raises(ValueError, match="geometry binding"):
        audit.compare_geometry(inputs, altered, force_branch)
