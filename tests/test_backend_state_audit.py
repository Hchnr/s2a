"""Decision and evidence-contract tests for the independent backend audit."""

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def audit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("backend_audit_test", root / "scripts/audit_wam_backend_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(position=100.0, rotation=0.5, opening=0.2):
    return {
        str(h): dict(wam=dict(position_rmse_mm=position, rotation_rmse_rad=rotation, opening_mae=opening))
        for h in (8, 16, 40)
    }


def test_backend_admission_needs_ten_percent_and_every_metric(audit):
    baseline = metrics()
    assert audit.eligibility(metrics(position=90), baseline)["passed"]
    assert not audit.eligibility(metrics(position=90.00000001), baseline)["passed"]
    assert not audit.eligibility(metrics(position=80, rotation=0.500000001), baseline)["passed"]
    assert not audit.eligibility(metrics(position=80, opening=0.200000001), baseline)["passed"]
    assert not audit.eligibility(metrics(position=0), metrics(position=0))["passed"]


def test_backend_nonregression_uses_primary_si_units(audit):
    baseline, candidate = metrics(), metrics(position=80)
    candidate["16"]["wam"]["position_rmse_mm"] = 100 + 5e-8
    result = audit.eligibility(candidate, baseline)
    assert result["passed"]
    assert not result["physical_trial_authorized"] and not result["fresh_evaluation_authorized"]
    candidate["16"]["wam"]["position_rmse_mm"] = 100 + 2e-7
    assert not audit.eligibility(candidate, baseline)["passed"]


def test_only_default_name_order_is_ignored_in_scheduler_identity(audit):
    # Diffusers serializes this one field using list(set(...)). The real
    # configuration and even the default-name set remain part of identity.
    baseline = dict(solver_order=2, shift=1, disable_corrector=[], _use_default_values=["solver_order", "shift"])
    reordered = dict(baseline, _use_default_values=["shift", "solver_order"])
    original = list(reordered["_use_default_values"])
    assert audit.scheduler_identity(reordered) == audit.scheduler_identity(baseline)
    assert reordered["_use_default_values"] == original
    for changed in (
        dict(baseline, solver_order=1),
        dict(baseline, shift=5),
        dict(baseline, disable_corrector=[1]),
        dict(baseline, _use_default_values=["shift"]),
        dict(baseline, unregistered_setting=True),
    ):
        assert audit.scheduler_identity(changed) != audit.scheduler_identity(baseline)


@pytest.mark.parametrize("names", [["shift", "shift"], ["not_a_setting"], [1], "shift"])
def test_scheduler_bookkeeping_cannot_hide_duplicate_or_unknown_keys(audit, names):
    with pytest.raises(ValueError, match="bookkeeping"):
        audit.scheduler_identity(dict(shift=1, _use_default_values=names))


def test_generated_state_distance_excludes_padded_future(audit):
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    reference = dict(
        generated=dict(position=[[0, 0, 0]] * 40, rotation=[identity] * 40, opening=[0.5] * 40),
        errors=dict(wam=dict(position=[0, 0])),
    )
    candidate = dict(
        generated=dict(
            position=[[0.003, 0.004, 0]] * 2 + [[1000, 0, 0]] * 38, rotation=[identity] * 40, opening=[0.25] * 40
        ),
        errors=dict(wam=dict(position=[0, 0])),
    )
    result = audit.state_distance([(reference, candidate)])
    for prefix in (8, 16, 40):
        assert result[str(prefix)]["valid_endpoints"] == 2
        assert result[str(prefix)]["position_rmse_mm"] == pytest.approx(5)
        assert result[str(prefix)]["rotation_rmse_rad"] == 0
        assert result[str(prefix)]["opening_mae"] == 0.25


def test_partial_backend_audit_cannot_generate_a_final_report(audit, tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("backend_report_test", root / "scripts/report_wam_backend_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pending = tmp_path / "partial.json"
    pending.write_text('{"complete":false,"status":"partial","errors":[]}')
    output = root / "outputs/never_create_backend_partial_report"
    target = root / "reports/never_create_backend_partial_report.md"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(
        "sys.argv", ["report", "--audit", str(pending), "--output-dir", str(output), "--report", str(target)]
    )
    with pytest.raises(ValueError, match="complete48"):
        module.main()
    assert not output.exists() and not target.exists()


def shutdown_fixture():
    names = [f"case{i}" for i in range(48)]
    artifact = dict(
        schema_version="s2a.readonly-shutdown-observability.v1",
        normal_owned_process_release=True,
        closed_json_exists=False,
        mask_evidence_json_exists=False,
        poisoned_final_memory_state_observed=None,
        initial_noise_rms_ledger_observed=None,
        completed_results_and_trace_pairs=48,
        cases=[dict(name=name) for name in names],
        new_model_requests=0,
        new_physics_steps=0,
    )
    probe = dict(
        cpu_only=True,
        new_model_requests=0,
        new_physics_steps=0,
        returncode=-15,
        entered_exists=True,
        finally_exists=False,
    )
    return names, artifact, probe


def test_narrow_shutdown_contract_preserves_unknown_memory_and_exact_coverage(audit):
    names, artifact, probe = shutdown_fixture()
    audit.validate_shutdown_supplement(artifact, probe, names)
    for field, value in (
        ("poisoned_final_memory_state_observed", False),
        ("initial_noise_rms_ledger_observed", []),
        ("completed_results_and_trace_pairs", 47),
        ("normal_owned_process_release", False),
        ("new_model_requests", 1),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        with pytest.raises(ValueError, match="supplement/probe identity"):
            audit.validate_shutdown_supplement(changed, probe, names)
    with pytest.raises(ValueError, match="supplement/probe identity"):
        audit.validate_shutdown_supplement(artifact, dict(probe, finally_exists=True), names)
    with pytest.raises(ValueError, match="supplement/probe identity"):
        audit.validate_shutdown_supplement(artifact, probe, names[::-1])


def test_missing_shutdown_receipts_still_fail_without_explicit_supplement(audit, tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "STUDY", tmp_path)
    args = SimpleNamespace(shutdown_supplement=None)
    with pytest.raises(FileNotFoundError, match="closed.json"):
        audit.shutdown_observability(audit.Evidence(), args, {}, {}, {})
