"""Check the independently reproduced endpoint mutation and its field mapping."""

import copy

import pytest

from scripts.simulation.runtime import causal_endpoint_diagnostic_admission as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.reference_ports import audit_causal_endpoint_diagnostic as audit


def test_actual_diagnostic_identifies_equivalent_object_quaternion():
    result = audit.load_audit()
    assert result["reproduced_original_failure"] and not result["control_qualified"]
    assert result["replayed_actions"] == 40 and result["task_success_tail_periods"] == 33
    assert result["changed_joint"]["name"] == "cream_cheese_1_joint0"
    assert all(event["rotation_difference_rad"] == 0 for event in result["events"])
    assert all(event["max_absolute_difference"] <= 3e-16 for event in result["events"])


def test_a_changed_velocity_cannot_be_classified_as_quaternion_normalization():
    inputs = admission.load_inputs()
    owner, branch = prefix.read(audit.OWNER), prefix.read(audit.BRANCH)
    branch = copy.deepcopy(branch)
    branch["endpoint_events"][1]["after"]["state"][-1] += 0.1
    with pytest.raises(ValueError, match="four quaternion components"):
        audit.summarize(inputs, owner, branch)
