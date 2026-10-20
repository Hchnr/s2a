"""Narrow exit observability never changes scientific checks or invents ledgers."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def audit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "sampler_release_audit_test", root / "scripts/audit_wam_sampler_steps_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log(pid=123, count=144):
    return "\n".join(
        [f"INFO: Started server process [{pid}]"]
        + ['INFO: 127.0.0.1 - "POST /generate HTTP/1.1" 200 OK'] * count
        + [
            "INFO: Shutting down",
            "INFO: Waiting for application shutdown.",
            "INFO: Application shutdown complete.",
            f"INFO: Finished server process [{pid}]",
        ]
    )


def test_all_quality_and_source_checks_are_literal_fork_of_frozen_auditor(audit):
    root = Path(__file__).resolve().parents[1]
    old, new = [
        ast.parse((root / "scripts" / n).read_text())
        for n in ("audit_wam_sampler_steps.py", "audit_wam_sampler_steps_release.py")
    ]
    old_functions = {n.name: ast.dump(n, include_attributes=False) for n in old.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: ast.dump(n, include_attributes=False) for n in new.body if isinstance(n, ast.FunctionDef)}
    for name in old_functions.keys() - {"main"}:
        assert new_functions[name] == old_functions[name], name


def test_exact_http_budget_and_owned_shutdown_order(audit):
    assert audit.check_shutdown_log(log(), 123)["successful_http_requests"] == 144
    for changed in (
        log(count=143),
        log().replace("200 OK", "500 ERROR", 1),
        log(pid=999),
        log() + '\n"POST /generate HTTP/1.1" 200 OK',
        log().replace("Application shutdown complete.", ""),
        '\n"POST /generate HTTP/1.1" 200 OK' + log(count=143),
    ):
        with pytest.raises(ValueError):
            audit.check_shutdown_log(changed, 123)


def test_missing_receipts_require_full_coverage_and_controlled_release(audit, tmp_path):
    for full, rc in ((False, -15), (True, -9), (True, 0)):
        with pytest.raises(ValueError, match="complete144"):
            audit.shutdown_observability(
                audit.Evidence(), tmp_path, dict(model_returncode=rc, last_job_returncode=0), 123, full_coverage=full
            )
    assert not (tmp_path / "service/closed.json").exists()
    assert not (tmp_path / "service/mask_evidence.json").exists()


def test_present_invalid_or_truncated_ledger_cannot_use_alternate_evidence(audit, tmp_path):
    service = tmp_path / "service"
    service.mkdir()
    close = service / "closed.json"
    for text in ('{"bad":true}', '{"counters":'):
        close.write_text(text)
        with pytest.raises(ValueError, match="ledger|receipt"):
            audit.shutdown_observability(
                audit.Evidence(), tmp_path, dict(model_returncode=-15, last_job_returncode=0), 123, full_coverage=True
            )


def test_complete_original_ledgers_remain_exact_and_observed(audit, tmp_path):
    service = tmp_path / "service"
    service.mkdir()
    (service / "closed.json").write_text(json.dumps(dict(counters={str(n): 48 for n in audit.STEPS}, poisoned=False)))
    (service / "mask_evidence.json").write_text(json.dumps([dict(inactive_rms_after=0)] * 144))
    result = audit.shutdown_observability(
        audit.Evidence(), tmp_path, dict(model_returncode=0, last_job_returncode=0), 123, full_coverage=True
    )
    assert result["exit_observability_complete"]
    assert result["final_in_memory_poisoned"] is False


def test_alternative_marks_memory_unknown_without_writing_receipts(audit, tmp_path, monkeypatch):
    (tmp_path / "orchestration/run").mkdir(parents=True)
    (tmp_path / "orchestration/run/service.log").write_text(log())
    review = tmp_path / "review.json"
    probe = tmp_path / "probe.json"
    review.write_text(json.dumps(dict(input_sha256={})))
    probe.write_text(
        json.dumps(
            dict(
                cpu_only=True,
                new_model_requests=0,
                new_physics_steps=0,
                returncode=-15,
                entered_exists=True,
                finally_exists=False,
                source_sha256={},
            )
        )
    )
    for label, path in (("EXIT_REVIEW", review), ("CPU_PROBE", probe)):
        monkeypatch.setattr(audit, label, path)
        monkeypatch.setattr(audit, label + "_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
    result = audit.shutdown_observability(
        audit.Evidence(), tmp_path, dict(model_returncode=-15, last_job_returncode=0), 123, full_coverage=True
    )
    assert result["exit_observability_complete"] is False
    assert result["initial_noise_rms_values"] is None
    assert result["final_in_memory_poisoned"] is None
    assert not (tmp_path / "service").exists()
