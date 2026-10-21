"""Only the documented non-control endpoint log difference is admissible."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture
def audit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "long_physical_audit_test", root / "scripts/audit_wam_long_physical_core.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows_with_only_known_difference(audit):
    old = [dict(step=i, command=[0.0], **{key: [0.1] for key in audit.DIAGNOSTICS}) for i in range(600)]
    new = copy.deepcopy(old) + [dict(step=i) for i in range(600, 1200)]
    for key in audit.DIAGNOSTICS:
        new[599][key] = [0.1 + 1e-16]
    return new, old


def test_exact_core_and_only_four_terminal_logs_are_separated(audit):
    new, old = rows_with_only_known_difference(audit)
    result = audit.physical_prefix(new, old)
    assert result["actual_state_command_motor_and_all_other_fields_exact"]
    assert not result["all600_shared_rows_exact"]
    assert len(result["logging_differences"]) == 4
    assert not result["new_scoring_tolerance_added"]


@pytest.mark.parametrize("change", ["control", "wrong_step", "large_diagnostic", "unknown_field"])
def test_any_new_physical_or_unregistered_difference_is_rejected(audit, change):
    new, old = rows_with_only_known_difference(audit)
    if change == "control":
        new[599]["command"] = [1e-16]
    elif change == "wrong_step":
        new[598]["raw_desired_tcp"] = [0.1 + 1e-16]
    elif change == "large_diagnostic":
        new[599]["raw_desired_tcp"] = [0.1 + 1e-9]
    else:
        new[599]["new_field"] = 1
    with pytest.raises(AssertionError):
        audit.physical_prefix(new, old)


def test_terminal_and_interior_branches_refer_to_the_same_endpoint(audit):
    targets = np.repeat(np.eye(4)[None], 40, axis=0)
    for i in range(40):
        targets[i, :3, :3] = Rotation.from_rotvec(np.array([0.13, -0.27, 0.19]) * (i + 1)).as_matrix()
        targets[i, :3, 3] = [0.01 * i, 0.0, 0.0]
    old = dict(raw_targets_tcp=targets[:8], anchor_tcp=np.eye(4))
    new = dict(raw_targets_tcp=targets, anchor_tcp=np.eye(4))
    old_pose, old_branch = audit.sample_diagnostic(old, "raw", 0.4, 0.05)
    new_pose, new_branch = audit.sample_diagnostic(new, "raw", 0.4, 0.05)
    assert old_branch == dict(targets=8, cursor=8.0, index=7, fraction=1.0)
    assert new_branch == dict(targets=40, cursor=8.0, index=8, fraction=0.0)
    np.testing.assert_array_equal(new_pose, targets[7])
    np.testing.assert_allclose(old_pose, targets[7], atol=1e-14, rtol=0)


def test_actual_source_consumes_diagnostic_pose_only_in_logged_metrics(audit, tmp_path):
    source = audit.REPO / "src/s2a/long_simulation.py"
    assert audit.diagnostic_only_source(source)["control_uses_actual_targets_index"]
    changed = tmp_path / "invalid_control.py"
    changed.write_text(
        source.read_text().replace(
            "converter.convert(measured, desired, twist)", "converter.convert(measured, raw_desired, twist)"
        )
    )
    with pytest.raises(AssertionError):
        audit.diagnostic_only_source(changed)


def test_diagnostic_reconstruction_reports_roundoff_and_rejects_existing_bound_violation(audit):
    targets = np.repeat(np.eye(4)[None], 40, axis=0)
    old = dict(step=592, raw_targets_tcp=targets[:8], effective_targets_tcp=targets[:8], anchor_tcp=np.eye(4))
    new = dict(step=592, raw_targets_tcp=targets, effective_targets_tcp=targets, anchor_tcp=np.eye(4))
    old_row = dict(step=599, model_time_s=0.4, after_source=np.eye(4).tolist())
    new_row = copy.deepcopy(old_row)
    for row in (old_row, new_row):
        for kind in ("raw", "effective"):
            row[f"{kind}_desired_tcp"] = np.eye(4).tolist()
            row[f"{kind}_model_tcp_error"] = np.zeros(6).tolist()
    old_row["raw_model_tcp_error"][0] = 1e-16
    result = audit.diagnostic_reconstruction(new_row, old_row, new, old, np.eye(4), 0.05)
    assert result["old"]["actual_reconstruction_residuals"]["raw"]["maximum_error_absolute_difference"] == 1e-16
    assert not result["old"]["actual_reconstruction_residuals"]["raw"]["exactly_equal"]
    assert result["old"]["all_four_logged_values_reconstructed_within_existing_tolerance"]
    old_row["raw_model_tcp_error"][0] = 2e-10
    with pytest.raises(AssertionError):
        audit.diagnostic_reconstruction(new_row, old_row, new, old, np.eye(4), 0.05)
