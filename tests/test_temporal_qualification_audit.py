"""Qualification must preserve canonical old fields, including tiny changes."""

import copy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def audit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "temporal_qualification_audit_test", root / "scripts/audit_wam_temporal_qualification.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def logs():
    old = [dict(step=i, desired=[0.0, 0.1], command=[0, -1], success=False) for i in range(600)]
    new = [
        dict(row, temporal_ensemble=dict(physical_step=i, contributors=[])) for i, row in enumerate(copy.deepcopy(old))
    ]
    return new, old


def test_only_new_provenance_is_excluded_from_old_fields(audit):
    new, old = logs()
    result = audit.common_row_identity(new, old)
    assert result["all600_common_step_fields_canonical_exact"]
    assert not result["numeric_tolerance_used_for_identity"]
    new[1]["extra_diagnostic"] = "not registered"
    with pytest.raises(AssertionError):
        audit.common_row_identity(new, old)


@pytest.mark.parametrize("value", [-0.0, 0, 1e-16])
def test_signed_zero_numeric_type_and_small_control_changes_fail_exact_gate(audit, value):
    new, old = logs()
    new[599]["desired"][0] = value
    result = audit.common_row_identity(new, old)
    assert not result["all600_common_step_fields_canonical_exact"]
    assert result["first_different_steps"] == [599] and result["different_step_count"] == 1


def test_oracle_launch_command_has_fixed_budget_no_model_entry(audit):
    command = audit.matrix_command()
    assert command[1].endswith("/scripts/run_wam_temporal_matrix.py")
    assert command[command.index("--source") + 1] == "oracle"
    assert command[command.index("--workers") + 1] == "2"
    assert "--checkpoint" not in command and "--job-indices" not in command
    assert audit.BUDGET == dict(oracle_trials=3, physical_steps=1800, model_requests=0, fresh_initial_states=0)
