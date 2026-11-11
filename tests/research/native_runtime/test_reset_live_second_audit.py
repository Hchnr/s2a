"""A model request needs its full actual denoise ledger to qualify as evidence."""

import copy

import pytest

from scripts.simulation.runtime import reset_live_second_admission as admission
from scripts.verification.response_models.audit_reset_live_second_window import validate_model_ledger


def test_model_audit_rejects_partial_denoise_and_wrong_device():
    model = dict(
        complete=True,
        passed=True,
        gpu_uuid=admission.GPU,
        model_initialization_attempts=1,
        model_initializations=1,
        model_requests_attempted=1,
        model_requests_completed=1,
        responses_sent=1,
        denoise_calls_attempted=35,
        denoise_calls_completed=35,
        denoise_evidence=[{"completed": True} for _ in range(35)],
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
        mask_evidence=[dict(stage="initial_noise", inactive_rms_before=0.1, inactive_rms_after=0.0)],
    )
    validate_model_ledger(model)
    bad = copy.deepcopy(model)
    bad["denoise_evidence"][7]["completed"] = False
    with pytest.raises(ValueError, match="35 actual denoise"):
        validate_model_ledger(bad)
    bad = copy.deepcopy(model)
    bad["gpu_uuid"] = "unregistered"
    with pytest.raises(ValueError, match="35 actual denoise"):
        validate_model_ledger(bad)


def test_imported_namespace_is_recorded_without_fabricating_a_file_hash(monkeypatch):
    from scripts.verification.response_models import audit_reset_live_second_window as audit

    imported = {
        "package": {"file": None, "namespace_package": True},
        "module": {"file": "/frozen/module.py", "sha256": "a" * 64},
    }
    monkeypatch.setattr(audit.repro, "sha256", lambda _path: "a" * 64)
    assert audit.validate_imported_sources(imported, {"/frozen/module.py": "a" * 64}) == (1, 1)
    with pytest.raises(ValueError, match="source manifest"):
        audit.validate_imported_sources(imported, {"/frozen/module.py": "b" * 64})


def test_second_window_keeps_absolute_step_alignment(monkeypatch):
    from scripts.verification.response_models import audit_reset_live_second_window as audit

    seen = []

    def comparator(left, right):
        seen.append(([row["step"] for row in left], [row["step"] for row in right]))
        return {"difference": 0}

    monkeypatch.setattr(audit, "compare_replay", comparator)
    rows = [{"step": i} for i in range(8, 16)]
    assert audit.compare_window(rows, rows, start_step=8) == {"difference": 0}
    assert seen == [(list(range(8)), list(range(8)))]
    bad = copy.deepcopy(rows)
    bad[0]["step"] = 0
    with pytest.raises(ValueError, match="absolute step"):
        audit.compare_window(rows, bad, start_step=8)
