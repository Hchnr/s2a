"""Verify the replay evidence and reject altered force timestamps or contacts."""

import copy

import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.reference_ports import audit_replayed_contact_torque as audit


def test_audit_rederives_original_replay_and_shared_force_history():
    result = audit.audit()
    assert result["shared_prefix_samples"] == 1050
    assert max(result["shared_prefix_max_difference"].values()) == 0
    assert result["shared_prefix_contacts_identical"] is True
    for name, count in (("historical", 1150), ("innovation", 1075)):
        branch = result["branches"][name]
        assert branch["continuation_samples"] == count
        assert max(branch["replay_max_difference"].values()) == 0
        row = next(item for item in branch["periods"] if item["step"] == 122)
        assert row["clipped_samples_by_joint"][6] == 25
        assert row["active_robot_environment_contact_samples"] == 25
    assert result["control_qualified"] is False and result["causal_mechanism_proven"] is False


def test_changed_observation_clock_and_shared_contact_are_rejected():
    inputs = audit.experiment.load_inputs()
    owner = prefix.read({"path": str(audit.RUN / "result.json"), "sha256": audit.OWNER_SHA})
    history = prefix.read(owner["summaries"]["historical"]["branch_ref"])
    innovation = prefix.read(owner["summaries"]["innovation"]["branch_ref"])
    damaged = copy.deepcopy(history)
    damaged["runtime"]["replay_dynamics"]["samples"][-1]["time_after"] += 0.001
    with pytest.raises(ValueError, match="align with actual replay"):
        audit.check_branch(inputs["replayed_contact_torque"]["branches"]["historical"]["producer"], damaged, 46)
    left = history["runtime"]["replay_dynamics"]["samples"]
    right = copy.deepcopy(innovation["runtime"]["replay_dynamics"]["samples"])
    right[0]["contacts"].append({"invalid": True})
    with pytest.raises(ValueError, match="contacts differ"):
        audit.check_shared_prefix(left, right)
